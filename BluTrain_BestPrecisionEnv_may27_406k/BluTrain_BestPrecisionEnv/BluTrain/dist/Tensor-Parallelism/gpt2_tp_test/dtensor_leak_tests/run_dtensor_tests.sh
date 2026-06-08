#!/bin/bash
set -e

cd "$(dirname "$0")/../.."

echo "Building main library..."
make all -j20

echo "Building DTensor leak tests..."
make dtensor_leak_tests -j20

echo "Running DTensor leak tests with 2 GPUs..."
if [ "$1" == "--valgrind" ]; then
    echo "Using Valgrind..."
    mpirun -np 2 valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes ./tests/dtensor_leak_tests/dtensor_leak_tests
else
    mpirun -np 2 ./tests/dtensor_leak_tests/dtensor_leak_tests
fi
