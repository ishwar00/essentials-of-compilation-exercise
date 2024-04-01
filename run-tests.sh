#!/bin/bash
export PYTHONPATH="$PWD:$PWD/interp_x86"
gcc -c -g -std=c99 runtime.c
python3 run-tests.py
