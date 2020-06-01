#!/bin/sh
#BSUB -q gpuv100
#BSUB -J Convnet
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 01:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -o ../outputs/gpu_%J.out
#BSUB -e ../outputs/gpu_%J.err
#BSUB -cwd "/zhome/27/c/138037/share/src/"
# -- end of LSF options --

nvidia-smi
# Load the cuda module
module load cuda/10.2

/appl/cuda/10.2/samples/NVIDIA_CUDA-10.2_Samples/bin/x86_64/linux/release/deviceQuery

module load python3
source ../venv/bin/activate
# Net - Epochs - Batchsizez - ScaleX - ScaleY - Droprate - N-feature
python3 main.py ConvNet 500 512 128 256 0.5 16 >| ../outputs/Convnet6.out 2>| ../outputs/Convnet6.err &&
python3 main.py ConvNet 500 512 128 256 0.5 8 >| ../outputs/Convnet7.out 2>| ../outputs/Convnet7.err &&
python3 main.py ConvNet 500 512 256 512 0.5 16 >| ../outputs/Convnet8.out 2>| ../outputs/Convnet8.err &&
python3 main.py ConvNet 500 512 256 512 0.5 8 >| ../outputs/Convnet9.out 2>| ../outputs/Convnet9.err

#python3 main.py ResNet 10 256 128 256 0.5 16 >| ../outputs/Resnet.out 2>| ../outputs/Resnet.err &&
#python3 main.py ResNet 10 256 64 128 0.5 16 >| ../outputs/Resnet2.out 2>| ../outputs/Resnet2.err


