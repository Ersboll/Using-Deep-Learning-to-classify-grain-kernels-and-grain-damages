#!/bin/sh
#BSUB -q gpuv100
#BSUB -J speedtest
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 01:30
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "select[gpu32gb]"
#BSUB -o ../outputs/gpu_%J.out
#BSUB -e ../outputs/gpu_%J.err
#BSUB -cwd "/zhome/e2/d/117429/foss/src/"
# -- end of LSF options --

nvidia-smi
# Load the cuda module
# module load cuda/10.2

# /appl/cuda/10.2/samples/NVIDIA_CUDA-10.2_Samples/bin/x86_64/linux/release/deviceQuery

module load python3
source ../env/bin/activate
# Model - Loss function - Epochs - Batchsize - Learning rate - Width - Height - Droprate - N-feature - Num_blocks - Intensity(on/off) - Transform (on/off) - Weighted sampling (on/off)
python3 main.py ConvNet crossentropy 25 256 0.01 64 128 0.6 64 0 0 0 0 >| ../outputs/Sampler0.out 2>| ../outputs/Sampler0.err

# python3 main.py ResNet focal 50 128 0.01 128 256 0.6 32 2 1 1 1>| ../outputs/ResNet2.out 2>| ../outputs/ResNet2.err

# python3 time_test.py ConvNet crossentropy 1 256 0.01 128 256 0.6 64 0 1 0 0>| ../outputs/Speedtest.out 2>| ../outputs/Speedtest.err
