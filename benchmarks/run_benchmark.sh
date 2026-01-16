#!/bin/bash
# QuantumTiler Benchmark Script
# Usage: ./run_benchmark.sh [size] [runs]

SIZE=${1:-2048}
RUNS=${2:-3}

echo "========================================"
echo "  QuantumTiler Benchmark"
echo "========================================"
echo "Matrix size: ${SIZE}x${SIZE}"
echo "Runs: ${RUNS}"
echo ""

# Build if needed
if [ ! -f "build/Release/quantum_tiler" ] && [ ! -f "build/Release/quantum_tiler.exe" ]; then
    echo "Building..."
    mkdir -p build && cd build
    cmake ..
    cmake --build . --config Release
    cd ..
fi

# Run benchmarks
echo ""
echo "Running standard benchmark..."
./build/Release/quantum_tiler $SIZE $SIZE $RUNS

echo ""
echo "Running stress mode benchmark..."
./build/Release/quantum_tiler $SIZE $SIZE $RUNS stress

echo ""
echo "Done! Results above."
