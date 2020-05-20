#!/bin/bash

#SBATCH --job-name="Emo_classf"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --ntasks-per-node=1
#SBATCH --time=2:00:00
#SBATCH --mem=8000M
#SBATCH --partition=gpu_shared_course
#SBATCH --gres=gpu:1

module purge

module load pre2019
module load Python/3.6.3-foss-2017b
module load cuDNN/7.0.5-CUDA-9.0.176
module load NCCL/2.0.5-CUDA-9.0.176
export LD_LIBRARY_PATH=/hpc/eb/Debian9/cuDNN/7.1-CUDA-8.0.44-GCCcore-5.4.0/lib64:$LD_LIBRARY_PATH

USER=`whoami`
WORKING_DIR=${TMPDIR}/${USER}
mkdir -p ${WORKING_DIR}
cp -r ~/atcs-project ${WORKING_DIR}/

cd ${WORKING_DIR}/atcs-project

pip install --user -r requirements.txt

srun python k_shot_testing.py --episodes "data/sentiment_episodes_k8.pkl" --num_updates 25 --model_path "0_Narrow_cls_512_256/best_test.pt"
