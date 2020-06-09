#!/bin/sh
#BSUB -q gpuv100
#BSUB -J ConvNet
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 03:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -o ../outputs/gpu_%J.out
#BSUB -e ../outputs/gpu_%J.err
###BSUB -cwd "Add own home/src directory here"
# -- end of LSF options --

nvidia-smi
# Load the cuda module
module load cuda/10.2

/appl/cuda/10.2/samples/NVIDIA_CUDA-10.2_Samples/bin/x86_64/linux/release/deviceQuery

module load python3
source ../venv/bin/activate
# Net - Epochs - Batchsizez - LR - ScaleX - ScaleY - Droprate - N-feature
python3 main.py ConvNet 500 512 0.01 128 256 0.5 16 >| ../outputs/Convnet0.out 2>| ../outputs/Convnet0.err
python3 main.py ConvNet 500 512 0.01 128 256 0.5 8 >| ../outputs/Convnet1.out 2>| ../outputs/Convnet1.err &&
#python3 main.py ConvNet 10 512 0.01 128 256 0.5 16 >| ../outputs/Convnet2.out 2>| ../outputs/Convnet2.err &&
#python3 main.py ConvNet 10 512 0.01 128 256 0.5 16 >| ../outputs/Convnet3.out 2>| ../outputs/Convnet3.err &&
#python3 main.py ConvNet 10 512 0.01 128 256 0.5 16 >| ../outputs/Convnet4.out 2>| ../outputs/Convnet4.err 

