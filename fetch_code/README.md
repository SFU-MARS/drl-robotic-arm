## Fetch environment tests

Scripts to run reinforcement learning training environments are stored here.
Instructions are written below.

### Requirements

Must have MuJoCo installed together with OpenAI Spinning Up. Refer to the links below:

- [OpenAI Spinning Up installation](https://spinningup.openai.com/en/latest/user/installation.html).
- [mujoco-py installation](https://github.com/openai/mujoco-py)

#### fetchPickAndPlace.py

In Linux, this script can be run directly from the bash script `fetch_init_linux.sh`.
Modify the initialization as needed (exported variables, virtual environment path,
etc.). If your computer is already set up to run the robot arm environment, one can
just run the Python script directly.

Modify the Python script directly to amend the observation wrapper, reward wrapper,
algorithm, hyperparameters.
