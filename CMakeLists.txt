cmake_minimum_required(VERSION 3.10)

# Project name and version
project(KhaosChess VERSION 2.18.0 LANGUAGES CXX)

# If not explicitly set, default to Release build type
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "No build type specified. Defaulting to Release.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

# Compiler settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)  # compile database for clangd
set(CMAKE_CXX_FLAGS "-g -Wall -Werror -Wextra -Wpedantic")

# Engine code as a library (everything except main.cpp)
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/*.cpp")
list(REMOVE_ITEM SOURCES "${CMAKE_SOURCE_DIR}/src/main.cpp")

add_library(khaos_core STATIC ${SOURCES})
target_include_directories(khaos_core PUBLIC "${CMAKE_SOURCE_DIR}/include")

find_package(Threads REQUIRED)
target_link_libraries(khaos_core PUBLIC Threads::Threads)

# Single source of truth for the engine version (reported over UCI)
target_compile_definitions(khaos_core PUBLIC VERSION="${PROJECT_VERSION}")

# The engine executable is now just main.cpp linked against the library
add_executable(${PROJECT_NAME} src/main.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE khaos_core)

# Score tuner (coordinate-descent and gradient modes)
add_executable(tuner tools/tuner.cpp)
target_link_libraries(tuner PRIVATE khaos_core tbb)

# Compiler specific settings
if (MINGW)
    target_compile_definitions(${PROJECT_NAME} PRIVATE WIN64 _AMD64_ IS_64BIT)
endif()

install(
  TARGETS ${PROJECT_NAME}
  RUNTIME DESTINATION bin)

# Show compiler and system information
message(STATUS "System: ${CMAKE_SYSTEM_NAME}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "Arch: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "Compiler flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# ------------------------- Tests -------------------------
enable_testing()

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
  DOWNLOAD_EXTRACT_TIMESTAMP true
)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

# Each test suite is its own binary in bin/tests/
function(add_khaos_test name)
  add_executable(${name} tests/${name}.cpp)
  target_link_libraries(${name} PRIVATE khaos_core GTest::gtest_main)
  set_target_properties(${name} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin/tests")
  gtest_discover_tests(${name})
endfunction()

add_khaos_test(position_tests)
add_khaos_test(perft_tests)
add_khaos_test(bitboard_tests)
add_khaos_test(draw_tests)
add_khaos_test(nnue_tests)
add_khaos_test(uci_tests)
