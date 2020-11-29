# Hindsight Experience Replay
For details on Hindsight Experience Replay (HER), please read the [paper](https://arxiv.org/abs/1707.01495).

Using pre-recorded demonstrations to Overcome the exploration problem in HER based Reinforcement learning.
For details, please read the [paper](https://arxiv.org/pdf/1709.10089.pdf).

### Getting started

To launch training with demonstrations (more technically, with behaviour cloning loss as an auxilliary loss), run the following
```bash
python -m baselines.run --alg=her --env=FetchPickAndPlace-v1 --num_timesteps=2.5e6 --demo_file=/Path/to/demo_file.npz
```
This will train a DDPG+HER agent on the `FetchPickAndPlace` environment by using previously generated demonstration data.
To inspect what the agent has learned, use the `--play` flag as described above.

demo_file is `data_fetch_random_100.npz`

### Configuration
The provided configuration is for training an agent with HER without demonstrations, we need to change a few paramters for the HER algorithm to learn through demonstrations, to do that, set:

* bc_loss: 1 - whether or not to use the behavior cloning loss as an auxilliary loss
* q_filter: 1 - whether or not a Q value filter should be used on the Actor outputs
* num_demo: 100 - number of expert demo episodes
* demo_batch_size: 128 - number of samples to be used from the demonstrations buffer, per mpi thread
* prm_loss_weight: 0.001 - Weight corresponding to the primary loss
* aux_loss_weight:  0.0078 - Weight corresponding to the auxilliary loss also called the cloning loss

Apart from these changes the reported results also have the following configurational changes:

* n_cycles: 20 - per epoch
* batch_size: 1024 - per mpi thread, total batch size
* random_eps: 0.1  - percentage of time a random action is taken
* noise_eps: 0.1  - std of gaussian noise added to not-completely-random actions

### Comment
Trial in the past:
```bash
python -m baselines.run --alg=her --env=FetchPickAndPlace-v1 --num_timesteps=2.5e5 --demo_file=/Path/to/demo_file.npz
```




