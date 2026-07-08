# Distributed GPU Matrix Operations Benchmark

This example measures the current distributed GPU implementation of the matrix
operations that dominate `pChASEGPU` grid-shape studies:

- two consecutive HEMM calls (`column -> row`, then `row -> column`)
- one device-buffer allreduce on the row communicator
- one device-buffer allreduce on the column communicator
- `DistMultiVector` redistribution from column to row and back to column

The benchmark can run those communication paths through GPU-aware MPI,
NCCL/RCCL, or both in the same executable via `--comm mpi|xccl|both`.
In addition to the small configurable allreduce payload, it also reports
allreduce timings for the actual HEMM result-buffer payloads in both
directions.

Both 1D block multivectors and 1D block-cyclic multivectors are checked for
correctness. HEMM timings are reported separately for the two directions, and
redistribution reports the generated segment counts for `column -> row` and
`row -> column`.

Example:

```bash
srun -n 128 --gpus-per-task=1 ./examples/7_matrix_operations_benchmark/matrix_operations_bench \
  --grid-rows 16 --grid-cols 8 \
  --n 20000 --nvec 80 --mb 256 --iters 20 --warmup 3 --comm both
```

ChASE currently requires `grid-rows >= grid-cols`. To compare rank mapping
without changing that shape constraint, keep the same legal grid and switch
the Cartesian ordering:

```bash
srun -n 8 --gpus-per-task=1 ./examples/7_matrix_operations_benchmark/matrix_operations_bench \
  --grid-rows 4 --grid-cols 2 --grid-major col

srun -n 8 --gpus-per-task=1 ./examples/7_matrix_operations_benchmark/matrix_operations_bench \
  --grid-rows 4 --grid-cols 2 --grid-major row
```
