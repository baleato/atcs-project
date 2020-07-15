#!/bin/bash

#SBATCH --job-name="k_shot"
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

model_path=~/results_ProtoMAML_Offenseval_mpl768_batch8_metabatch32_lr1e-4_ilr1e-3_unfreeze12_numiter3000/best_test_Offenseval_acc_0.66406_iter_400_model.pt
model=ProtoMAML #MTL ProtoNet
echo ${model}: ${model_path}
validation_task=Offenseval
for task in Abuse
do
	for k in 4 8 16
	do
		for number in 5 25 100 300
		do
			echo =================== ${model} val:${validation_task} - test:${task} - ${k}-shot - ${number} updates ====================
			srun python k_shot_testing.py \
				--model_path ${model_path} \
				--batch_size 32 \
				--mlp_dims 768 \
				--lr 0.001 \
				--model ${model} --task ${task} --k ${k} --num_updates ${number} \
				--episodes data/episodes/episodes_${task}_k_${k}_episodes_32.pkl
		done
	done
done
