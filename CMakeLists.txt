cmake_minimum_required(VERSION 3.10)
project(QuantumTiler 
    VERSION 1.0.0
    DESCRIPTION "Quantum-inspired adaptive tiling for matrix multiplication"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(ENABLE_OPENMP "Enable OpenMP parallelization" ON)
option(ENABLE_AVX2 "Enable AVX2/FMA instructions" ON)
option(BUILD_TESTS "Build test executables" OFF)

# Detect compiler and set flags
if(MSVC)
    # MSVC flags
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2 /EHsc")
    if(ENABLE_AVX2)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2")
    endif()
    if(ENABLE_OPENMP)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp")
    endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # GCC/Clang flags
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra")
    if(ENABLE_AVX2)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma")
    endif()
    if(ENABLE_OPENMP)
        find_package(OpenMP)
        if(OpenMP_CXX_FOUND)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        endif()
    endif()
    # Native tuning for best performance on local CPU
    option(ENABLE_NATIVE "Enable -march=native for best local performance" OFF)
    if(ENABLE_NATIVE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    endif()
endif()

# Source files
set(QUANTUM_TILER_SRC
    ${CMAKE_SOURCE_DIR}/src/quantum_tiler.cpp
)

# Fallback to root if src/ doesn't exist
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/src/quantum_tiler.cpp")
    if(EXISTS "${CMAKE_SOURCE_DIR}/adaptive_simple.cpp")
        set(QUANTUM_TILER_SRC ${CMAKE_SOURCE_DIR}/adaptive_simple.cpp)
    endif()
endif()

# Main executable
add_executable(quantum_tiler ${QUANTUM_TILER_SRC})

# Platform-specific libraries
if(WIN32)
    target_link_libraries(quantum_tiler PRIVATE pdh)
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(quantum_tiler PRIVATE OpenMP::OpenMP_CXX)
endif()

# Install rules
install(TARGETS quantum_tiler
    RUNTIME DESTINATION bin
)

install(FILES README.md LICENSE
    DESTINATION share/doc/QuantumTiler
)

# Print configuration summary
message(STATUS "")
message(STATUS "=== QuantumTiler Configuration ===")
message(STATUS "Version:     ${PROJECT_VERSION}")
message(STATUS "Compiler:    ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "Build type:  ${CMAKE_BUILD_TYPE}")
message(STATUS "OpenMP:      ${ENABLE_OPENMP}")
message(STATUS "AVX2/FMA:    ${ENABLE_AVX2}")
message(STATUS "CXX Flags:   ${CMAKE_CXX_FLAGS}")
message(STATUS "================================")
message(STATUS "")
