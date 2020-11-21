#!/bin/bash

gnome-terminal & disown
cd ~/spinningup

# activate python3 environment
. ~/pyenv/aigym/venv/bin/activate

# config mujoco environment
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/dingyi/.mujoco/mujoco200/bin
