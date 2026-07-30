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
    const std::int16_t* own = acc.v[stm];
    const std::int16_t* their = acc.v[~stm];

    // Clipped ReLU to [0, QA], then the int16 x int16 -> int32 output layer.
    // Widest intermediate is 2 * HIDDEN * QA * max|W2|, comfortably inside
    // int32 for any sanely trained net.
    std::int32_t sum = 0;

    for (std::int32_t i = 0; i < HIDDEN; ++i) {
        const std::int32_t a = std::clamp<std::int32_t>(own[i], 0, QA);
        sum += a * static_cast<std::int32_t>(net->output_weights[i]);
    }
    for (std::int32_t i = 0; i < HIDDEN; ++i) {
        const std::int32_t a = std::clamp<std::int32_t>(their[i], 0, QA);
        sum += a * static_cast<std::int32_t>(net->output_weights[HIDDEN + i]);
    }

    sum += net->output_bias;

    // Back to engine eval units. int64 because sum * EVAL_SCALE overflows
    // int32 well before the accumulator itself does.
    const std::int64_t scaled = static_cast<std::int64_t>(sum) * EVAL_SCALE;
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
    std::int32_t qa = 0;
    std::int32_t qb = 0;
    std::int32_t eval_scale = 0;
    std::int32_t reserved[4];

    if (!read_scalar(in, version) || !read_scalar(in, inputs) ||
        !read_scalar(in, hidden) || !read_scalar(in, qa) ||
        !read_scalar(in, qb) || !read_scalar(in, eval_scale) ||
        !read_array(in, reserved, 4)) {
        error = "truncated header";
        return false;
    }

    if (version != FORMAT_VERSION) {
        error = "net format version " + std::to_string(version) +
                ", engine expects " + std::to_string(FORMAT_VERSION);
        return false;
    }
    if (inputs != static_cast<std::uint32_t>(INPUTS) ||
        hidden != static_cast<std::uint32_t>(HIDDEN)) {
        error = "topology " + std::to_string(inputs) + "x" +
                std::to_string(hidden) + ", engine is built for " +
                std::to_string(INPUTS) + "x" + std::to_string(HIDDEN);
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
        !read_array(in, candidate->output_weights, 2 * HIDDEN) ||
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

    const std::uint32_t inputs = INPUTS;
    const std::uint32_t hidden = HIDDEN;
    const std::int32_t qa = QA;
    const std::int32_t qb = QB;
    const std::int32_t eval_scale = EVAL_SCALE;
    const std::int32_t reserved[4] = {0, 0, 0, 0};
    const std::uint32_t version = FORMAT_VERSION;

    out.write(MAGIC, 8);
    out.write(reinterpret_cast<const char*>(&version), sizeof(version));
    out.write(reinterpret_cast<const char*>(&inputs), sizeof(inputs));
    out.write(reinterpret_cast<const char*>(&hidden), sizeof(hidden));
    out.write(reinterpret_cast<const char*>(&qa), sizeof(qa));
    out.write(reinterpret_cast<const char*>(&qb), sizeof(qb));
    out.write(reinterpret_cast<const char*>(&eval_scale), sizeof(eval_scale));
    out.write(reinterpret_cast<const char*>(reserved), sizeof(reserved));

    std::uint64_t state = seed;

    // Keep feature weights small: 32 pieces x 2 perspectives must not push the
    // int16 accumulator anywhere near saturation, which is the same constraint
    // the trainer enforces by clamping.
    auto next_feature_weight = [&state]() -> std::int16_t {
        return static_cast<std::int16_t>(
            static_cast<std::int32_t>(splitmix64(state) % 129) - 64);
    };

    std::vector<std::int16_t> feature_weights(
        static_cast<std::size_t>(INPUTS) * HIDDEN);
    for (auto& w : feature_weights) {
        w = next_feature_weight();
    }
    out.write(reinterpret_cast<const char*>(feature_weights.data()),
              static_cast<std::streamsize>(feature_weights.size() *
                                           sizeof(std::int16_t)));

    std::vector<std::int16_t> feature_bias(HIDDEN);
    for (auto& b : feature_bias) {
        b = next_feature_weight();
    }
    out.write(reinterpret_cast<const char*>(feature_bias.data()),
              static_cast<std::streamsize>(feature_bias.size() *
                                           sizeof(std::int16_t)));

    std::vector<std::int16_t> output_weights(2 * HIDDEN);
    for (auto& w : output_weights) {
        w = static_cast<std::int16_t>(
            static_cast<std::int32_t>(splitmix64(state) % 65) - 32);
    }
    out.write(reinterpret_cast<const char*>(output_weights.data()),
              static_cast<std::streamsize>(output_weights.size() *
                                           sizeof(std::int16_t)));

    const std::int32_t output_bias = 0;
    out.write(reinterpret_cast<const char*>(&output_bias), sizeof(output_bias));

    return static_cast<bool>(out);
}

}  // namespace nnue
}  // namespace KhaosChess
