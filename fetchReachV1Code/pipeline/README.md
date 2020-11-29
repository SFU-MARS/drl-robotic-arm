#### Note

This folder will contain the code to run several pre-trained models at different phases in the pick and place task.

It will reference other models stored in other folders, so no need to move it all here.

## Fetch environment tests

Scripts to run reinforcement learning training environments are stored here.
Instructions are written below.

### Requirements

Must have MuJoCo installed together with OpenAI Spinning Up. Refer to the links below:

- [OpenAI Spinning Up installation](https://spinningup.openai.com/en/latest/user/installation.html).
- [mujoco-py installation](https://github.com/openai/mujoco-py)

#### test_policy.py

This file must be copied into `spinningup/spinup/utils`; it contains a new function
`run_pipeline` which is used in `pipeline.py`.

#### pipeline.py

In Linux, this script can be run directly from the bash script `fetch_init_linux.sh`.
Modify the initialization as needed (exported variables, virtual environment path,
etc.). If your computer is already set up to run the robot arm environment, one can
just run the Python script directly.

The paths in `pipeline.py` are hardcoded, so this file must be executed in the current
directory (this is already done in `fetch_init_linux.sh`).
