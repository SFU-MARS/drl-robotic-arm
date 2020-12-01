python -m spinup.run ppo --hid "[128, 256, 128]" --env FetchReach-v1 --exp_name frv1test --steps_per_epoch 12000 --gamma 0.99 --epochs 200 --cpu 12 --pi_lr 3e-4 --vf_lr 3e-4
