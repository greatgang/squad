#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters"
    exit 1
fi

export EVAL_EXP="$1"

cl work main::cs224n-GreatGang
cl upload code
cl upload experiments/$EVAL_EXP/best_checkpoint

