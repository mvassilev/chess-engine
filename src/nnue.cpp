#include "nnue.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <memory>
#include <vector>

namespace KhaosChess {
namespace nnue {

const Network* net = nullptr;
bool use_nnue = true;  // no effect until a net is loaded

namespace {

// Owns whatever `net` points at. Swapped only between searches.
std::unique_ptr<Network> loaded_net;

template <typename T>
bool read_scalar(std::istream& in, T& out) {
    in.read(reinterpret_cast<char*>(&out), sizeof(T));
    return static_cast<bool>(in);
}

template <typename T>
bool read_array(std::istream& in, T* out, std::size_t count) {
    in.read(reinterpret_cast<char*>(out), sizeof(T) * count);
    return static_cast<bool>(in);
}

// splitmix64: a tiny, self-contained PRNG so the test net is reproducible from
// a seed without pulling in the engine's Zobrist generator.
std::uint64_t splitmix64(std::uint64_t& state) {
    std::uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

}  // namespace

Value forward(const Accumulator& acc, Color stm) {
    // Concatenated clipped activations: own half, then their half.
    std::int32_t x[2 * HIDDEN];
    for (std::int32_t i = 0; i < HIDDEN; ++i) {
        x[i] = std::clamp<std::int32_t>(acc.v[stm][i], 0, QA);
        x[HIDDEN + i] = std::clamp<std::int32_t>(acc.v[~stm][i], 0, QA);
    }

    // Hidden layer 2: /QB returns to the QA activation scale, then clip.
    std::int32_t h[L2];
    for (std::int32_t j = 0; j < L2; ++j) {
        std::int32_t sum = net->l2_bias[j];
        const std::int16_t* row = net->l2_weights[j];
        for (std::int32_t i = 0; i < 2 * HIDDEN; ++i) {
            sum += x[i] * static_cast<std::int32_t>(row[i]);
        }
        h[j] = std::clamp<std::int32_t>(sum / QB, 0, QA);
    }

    // Output layer, then back to engine eval units (int64: out * EVAL_SCALE
    // overflows int32).
    std::int32_t out = net->output_bias;
    for (std::int32_t j = 0; j < L2; ++j) {
        out += h[j] * static_cast<std::int32_t>(net->output_weights[j]);
    }

    const std::int64_t scaled = static_cast<std::int64_t>(out) * EVAL_SCALE;
    return static_cast<Value>(scaled / (static_cast<std::int64_t>(QA) * QB));
}

bool load(const std::string& path, std::string& error) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        error = "cannot open '" + path + "'";
        return false;
    }

    char magic[8];
    if (!read_array(in, magic, 8) || std::memcmp(magic, MAGIC, 8) != 0) {
        error = "'" + path + "' is not a KhaosChess net (bad magic)";
        return false;
    }

    std::uint32_t version = 0;
    std::uint32_t inputs = 0;
    std::uint32_t hidden = 0;
    std::uint32_t l2 = 0;
    std::int32_t qa = 0;
    std::int32_t qb = 0;
    std::int32_t eval_scale = 0;
    std::int32_t reserved[3];

    if (!read_scalar(in, version) || !read_scalar(in, inputs) ||
        !read_scalar(in, hidden) || !read_scalar(in, l2) ||
        !read_scalar(in, qa) || !read_scalar(in, qb) ||
        !read_scalar(in, eval_scale) || !read_array(in, reserved, 3)) {
        error = "truncated header";
        return false;
    }

    if (version != FORMAT_VERSION) {
        error = "net format version " + std::to_string(version) +
                ", engine expects " + std::to_string(FORMAT_VERSION);
        return false;
    }
    if (inputs != static_cast<std::uint32_t>(INPUTS) ||
        hidden != static_cast<std::uint32_t>(HIDDEN) ||
        l2 != static_cast<std::uint32_t>(L2)) {
        error = "topology " + std::to_string(inputs) + "x" +
                std::to_string(hidden) + "x" + std::to_string(l2) +
                ", engine is built for " + std::to_string(INPUTS) + "x" +
                std::to_string(HIDDEN) + "x" + std::to_string(L2);
        return false;
    }
    // A mismatch here would silently mis-scale every evaluation, so it is a
    // hard error rather than something to adapt to at runtime.
    if (qa != QA || qb != QB || eval_scale != EVAL_SCALE) {
        error = "quantization mismatch (file QA/QB/scale " + std::to_string(qa) +
                "/" + std::to_string(qb) + "/" + std::to_string(eval_scale) +
                ", engine " + std::to_string(QA) + "/" + std::to_string(QB) +
                "/" + std::to_string(EVAL_SCALE) + ")";
        return false;
    }

    auto candidate = std::make_unique<Network>();

    if (!read_array(in, &candidate->feature_weights[0][0],
                    static_cast<std::size_t>(INPUTS) * HIDDEN) ||
        !read_array(in, candidate->feature_bias, HIDDEN) ||
        !read_array(in, &candidate->l2_weights[0][0],
                    static_cast<std::size_t>(L2) * 2 * HIDDEN) ||
        !read_array(in, candidate->l2_bias, L2) ||
        !read_array(in, candidate->output_weights, L2) ||
        !read_scalar(in, candidate->output_bias)) {
        error = "truncated weights";
        return false;
    }

    // Trailing bytes mean the writer and reader disagree about the layout even
    // though the header matched, so refuse rather than evaluate with garbage.
    in.peek();
    if (!in.eof()) {
        error = "trailing bytes after weights";
        return false;
    }

    loaded_net = std::move(candidate);
    net = loaded_net.get();
    return true;
}

void unload() {
    net = nullptr;
    loaded_net.reset();
}

bool write_random_net(const std::string& path, std::uint64_t seed) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        return false;
    }

    const std::uint32_t version = FORMAT_VERSION, inputs = INPUTS,
                        hidden = HIDDEN, l2 = L2;
    const std::int32_t qa = QA, qb = QB, eval_scale = EVAL_SCALE;
    const std::int32_t reserved[3] = {0, 0, 0};

    out.write(MAGIC, 8);
    out.write(reinterpret_cast<const char*>(&version), sizeof(version));
    out.write(reinterpret_cast<const char*>(&inputs), sizeof(inputs));
    out.write(reinterpret_cast<const char*>(&hidden), sizeof(hidden));
    out.write(reinterpret_cast<const char*>(&l2), sizeof(l2));
    out.write(reinterpret_cast<const char*>(&qa), sizeof(qa));
    out.write(reinterpret_cast<const char*>(&qb), sizeof(qb));
    out.write(reinterpret_cast<const char*>(&eval_scale), sizeof(eval_scale));
    out.write(reinterpret_cast<const char*>(reserved), sizeof(reserved));

    std::uint64_t state = seed;
    auto rnd = [&state](std::int32_t range) -> std::int32_t {
        return static_cast<std::int32_t>(splitmix64(state) % (2 * range + 1)) -
               range;
    };
    auto dump16 = [&out](const std::vector<std::int16_t>& v) {
        out.write(reinterpret_cast<const char*>(v.data()),
                  static_cast<std::streamsize>(v.size() * sizeof(std::int16_t)));
    };

    // Small feature weights so 32 pieces cannot saturate the int16 accumulator.
    std::vector<std::int16_t> feature_weights(
        static_cast<std::size_t>(INPUTS) * HIDDEN);
    for (auto& w : feature_weights) w = static_cast<std::int16_t>(rnd(64));
    dump16(feature_weights);

    std::vector<std::int16_t> feature_bias(HIDDEN);
    for (auto& b : feature_bias) b = static_cast<std::int16_t>(rnd(64));
    dump16(feature_bias);

    std::vector<std::int16_t> l2_weights(static_cast<std::size_t>(L2) * 2 *
                                         HIDDEN);
    for (auto& w : l2_weights) w = static_cast<std::int16_t>(rnd(64));
    dump16(l2_weights);

    for (std::int32_t j = 0; j < L2; ++j) {
        const std::int32_t b = rnd(128);
        out.write(reinterpret_cast<const char*>(&b), sizeof(b));
    }

    std::vector<std::int16_t> output_weights(L2);
    for (auto& w : output_weights) w = static_cast<std::int16_t>(rnd(32));
    dump16(output_weights);

    const std::int32_t output_bias = 0;
    out.write(reinterpret_cast<const char*>(&output_bias), sizeof(output_bias));

    return static_cast<bool>(out);
}

}  // namespace nnue
}  // namespace KhaosChess
