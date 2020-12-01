# Train a PPO with gym environment FetchReach-V1

Source code based on the OpenAI Spinning Up repo to Train a PPO network under the gym FetchReach-V1 environment. 

## Requirements

- OpenAI Spinning Up
- mujoco 2.0

## How to run

1. Replace the source code `/<dir to your local spinningup repo >/spinningup/spinup/algos/pytorch/ppo/ppo.py` with the `ppo.py` under this repo. 
2. To train with gripper enabled, replace the source code `/<dir to your gym libriay>/gym/envs/robotics/fetch/reach.py` with the `reach.py` under this repo.
3. Configure essential environment with `source aigym_spinningup.sh`. You should change the paths in file `aigym_spinningup.sh` based on your system and installation. 
4. Run the command `python -m spinup.run ppo --hid "[256, 256, 256]" --env FetchReach-v1 --exp_name frv1test --steps_per_epoch 12000 --gamma 0.999 --epochs 100 --cpu 12 --pi_lr 3e-4 --vf_lr 3e-4` under the spinningup directory.

## Evaluate pre-trained network

Run the command `python -m spinup.run test_policy <dir to /pretrainedNetwork> --itr 90` under the spinningup directory.

## Models
