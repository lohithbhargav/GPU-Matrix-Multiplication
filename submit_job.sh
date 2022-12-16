#!/bin/bash
#SBATCH --job-name=CS5375_ldoppala
#SBATCH --output=%j.%N.out
#SBATCH --error=%j.%N.err
#SBATCH --partition=matador
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --account=cs5375
#SBATCH --reservation=cs5375_gpu

module load gcc cuda

nvcc matrixMul_gpu_v2.cu -o matrixMul_gpu_v2.exe
nvprof ./matrixMul_gpu_v2.exe