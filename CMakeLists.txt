cmake_minimum_required(VERSION 3.10)
project(QuantumTiler CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable optimizations
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2 /arch:AVX2 /openmp")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mavx2 -mfma -fopenmp")
endif()

# Main executable - can build from either location
if(EXISTS "${CMAKE_SOURCE_DIR}/src/quantum_tiler.cpp")
    add_executable(quantum_tiler src/quantum_tiler.cpp)
else()
    add_executable(quantum_tiler adaptive_simple.cpp)
endif()

if(WIN32)
    target_link_libraries(quantum_tiler pdh)
endif()
