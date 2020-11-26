#!/bin/bash
#
# script to run OpenAI environments with MuJoCo
# variables should be tailored to your system setup

export LD_LIBRARY_PATH="/home/max/.mujoco/mujoco200/bin"
export LD_PRELOAD="/usr/lib/libGLEW.so:/usr/lib/libGL.so"

SCRIPT_DIR="$(dirname $0)"
VENV_DIR=""
PYTHON_CMD="python3.7"

if [ -z "$VENV_DIR" ]; then
    echo "Please set the virtual environment directory in this script." 
    exit 1
fi

. $VENV_DIR/bin/activate

cd $SCRIPT_DIR

$PYTHON_CMD pipeline.py
