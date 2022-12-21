#!/bin/bash

seed='1 2 3 4'
seed_last='5'
#for seed_index in $seed
#do
#python main.py --env-name "HalfCheetah-v2" --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 5000000 --use-linear-lr-decay --use-proper-time-limits --gail --seed $seed_index --vis_name mujoco_baseline_$seed_index --no-cuda &
#python main.py --env-name "HalfCheetah-v2" --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 5000000 --use-linear-lr-decay --use-proper-time-limits --gail --seed $seed_index --vis_name mujoco_gail_reward_first_$seed_index --gail_reward first --no-cuda &
#done
#python main.py --env-name "HalfCheetah-v2" --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 5000000 --use-linear-lr-decay --use-proper-time-limits --gail --seed $seed_last --vis_name mujoco_baseline_$seed_last --no-cuda &
#python main.py --env-name "HalfCheetah-v2" --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 5000000 --use-linear-lr-decay --use-proper-time-limits --gail --seed $seed_last --vis_name mujoco_gail_reward_first_$seed_last --gail_reward first --no-cuda
#
#for seed_index in $seed
#do
#python main.py --env-name "HalfCheetah-v2" --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 5000000 --use-linear-lr-decay --use-proper-time-limits --gail --seed $seed_index --vis_name mujoco_gail_reward_second_$seed_index --gail_reward second --no-cuda &
#python main.py --env-name "HalfCheetah-v2" --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 5000000 --use-linear-lr-decay --use-proper-time-limits --gail --seed $seed_index --vis_name mujoco_11_expert_$seed_index --num_expert_data 11 --no-cuda &
#done
#python main.py --env-name "HalfCheetah-v2" --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 5000000 --use-linear-lr-decay --use-proper-time-limits --gail --seed $seed_last --vis_name mujoco_gail_reward_second_$seed_last --gail_reward second --no-cuda &
#python main.py --env-name "HalfCheetah-v2" --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 5000000 --use-linear-lr-decay --use-proper-time-limits --gail --seed $seed_last --vis_name mujoco_11_expert_$seed_last --num_expert_data 11 --no-cuda
#
#
#for seed_index in $seed
#do
##python main.py --env-name "HalfCheetah-v2" --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 5000000 --use-linear-lr-decay --use-proper-time-limits --gail --seed $seed_index --vis_name mujoco_25_expert_$seed_index --num_expert_data 25 --no-cuda &
#python main.py --env-name "HalfCheetah-v2" --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 5000000 --use-linear-lr-decay --use-proper-time-limits --gail --seed $seed_index --vis_name mujoco_no_rms_$seed_index --no_rms --no-cuda &
#done
##python main.py --env-name "HalfCheetah-v2" --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 5000000 --use-linear-lr-decay --use-proper-time-limits --gail --seed $seed_last --vis_name mujoco_25_expert_$seed_last --num_expert_data 25 --no-cuda &
#python main.py --env-name "HalfCheetah-v2" --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 5000000 --use-linear-lr-decay --use-proper-time-limits --gail --seed $seed_last --vis_name mujoco_no_rms_$seed_last --no_rms --no-cuda
#
#
#for seed_index in $seed
#do
#python main.py --env-name "HalfCheetah-v2" --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 5000000 --use-linear-lr-decay --use-proper-time-limits --gail --seed $seed_index --vis_name mujoco_2_epoch_$seed_index --gail-epoch 2 --no-cuda &
#python main.py --env-name "HalfCheetah-v2" --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 5000000 --use-linear-lr-decay --use-proper-time-limits --gail --seed $seed_index --vis_name mujoco_8_epoch_$seed_index --gail-epoch 8 --no-cuda &
#done
#python main.py --env-name "HalfCheetah-v2" --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 5000000 --use-linear-lr-decay --use-proper-time-limits --gail --seed $seed_last --vis_name mujoco_2_epoch_$seed_last --gail-epoch 2 --no-cuda &
#python main.py --env-name "HalfCheetah-v2" --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 5000000 --use-linear-lr-decay --use-proper-time-limits --gail --seed $seed_last --vis_name mujoco_8_epoch_$seed_last --gail-epoch 8 --no-cuda
#
#
#for seed_index in $seed
#do
#python main.py --env-name "HalfCheetah-v2" --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 5000000 --use-linear-lr-decay --use-proper-time-limits --gail --seed $seed_index --vis_name mujoco_16_epoch_$seed_index --gail-epoch 16 --no-cuda &
#python main.py --env-name "HalfCheetah-v2" --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 --lr 1e-5 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 5000000 --use-linear-lr-decay --use-proper-time-limits --gail --seed $seed_index --vis_name mujoco_lr_1e-5_$seed_index --warm_start_epoch 1 --gail-epoch 2 --no-cuda &
#done
#python main.py --env-name "HalfCheetah-v2" --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 5000000 --use-linear-lr-decay --use-proper-time-limits --gail --seed $seed_last --vis_name mujoco_16_epoch_$seed_last --gail-epoch 16 --no-cuda &
#python main.py --env-name "HalfCheetah-v2" --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 --lr 1e-5 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 5000000 --use-linear-lr-decay --use-proper-time-limits --gail --seed $seed_last --vis_name mujoco_lr_1e-5_$seed_last --warm_start_epoch 1 --gail-epoch 2 --no-cuda
#
#
#for seed_index in $seed
#do
#python main.py --env-name "HalfCheetah-v2" --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 --lr 5e-5 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 5000000 --use-linear-lr-decay --use-proper-time-limits --gail --seed $seed_index --vis_name mujoco_lr_5e-5_$seed_index --warm_start_epoch 1 --gail-epoch 2 --no-cuda &
#python main.py --env-name "HalfCheetah-v2" --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 --lr 1e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 5000000 --use-linear-lr-decay --use-proper-time-limits --gail --seed $seed_index --vis_name mujoco_lr_1e-4_$seed_index --warm_start_epoch 1 --gail-epoch 2 --no-cuda &
#done
#python main.py --env-name "HalfCheetah-v2" --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 --lr 5e-5 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 5000000 --use-linear-lr-decay --use-proper-time-limits --gail --seed $seed_last --vis_name mujoco_lr_5e-5_$seed_last --warm_start_epoch 1 --gail-epoch 2 --no-cuda &
#python main.py --env-name "HalfCheetah-v2" --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 --lr 1e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 5000000 --use-linear-lr-decay --use-proper-time-limits --gail --seed $seed_last --vis_name mujoco_lr_1e-4_$seed_last --warm_start_epoch 1 --gail-epoch 2 --no-cuda
#
#for seed_index in $seed
#do
#python main.py --env-name "HalfCheetah-v2" --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 5000000 --use-linear-lr-decay --use-proper-time-limits --gail --seed $seed_index --vis_name mujoco_lr_3e-4_$seed_index --warm_start_epoch 1 --gail-epoch 2 --no-cuda &
#python main.py --env-name "HalfCheetah-v2" --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 5000000 --use-linear-lr-decay --use-proper-time-limits --gail --seed $seed_index --vis_name mujoco_no_extra_loss_$seed_index --gail_loss no_extra_loss --no-cuda &
#done
#python main.py --env-name "HalfCheetah-v2" --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 5000000 --use-linear-lr-decay --use-proper-time-limits --gail --seed $seed_last --vis_name mujoco_lr_3e-4_$seed_last --warm_start_epoch 1 --gail-epoch 2 --no-cuda &
#python main.py --env-name "HalfCheetah-v2" --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 5000000 --use-linear-lr-decay --use-proper-time-limits --gail --seed $seed_last --vis_name mujoco_no_extra_loss_$seed_last --gail_loss no_extra_loss --no-cuda
#
python main.py --env-name "HalfCheetah-v2" --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 2 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 5000000 --use-linear-lr-decay --use-proper-time-limits --gail --seed 1 --vis_name mujoco_new_1 --no-cuda &
python main.py --env-name "HalfCheetah-v2" --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 5 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 5000000 --use-linear-lr-decay --use-proper-time-limits --gail --seed 1 --vis_name mujoco_new_2 --no-cuda &
python main.py --env-name "HalfCheetah-v2" --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 5000000 --use-linear-lr-decay --use-proper-time-limits --gail --seed 1 --vis_name mujoco_new_3 --no-cuda &
#
python main.py --env-name "HalfCheetah-v2" --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 2 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 5000000 --use-linear-lr-decay --use-proper-time-limits --gail --seed 1 --vis_name mujoco_new_4 --no-cuda --gail-batch-size 32 &
python main.py --env-name "HalfCheetah-v2" --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 5 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 5000000 --use-linear-lr-decay --use-proper-time-limits --gail --seed 1 --vis_name mujoco_new_5 --no-cuda --gail-batch-size 32 &
python main.py --env-name "HalfCheetah-v2" --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes 1 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 5000000 --use-linear-lr-decay --use-proper-time-limits --gail --seed 1 --vis_name mujoco_new_6 --no-cuda --gail-batch-size 32 &