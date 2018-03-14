#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters"
    exit 1
fi

export EVAL_EXP="$1"
export OUT_FILE_NAME="${EVAL_EXP}_preds.json"

echo "Generating json file: $OUT_FILE_NAME"

python code/main.py --mode=official_eval --json_in_path=data/tiny-dev.json --json_out_path="eval_data/${OUT_FILE_NAME}" --ckpt_load_dir="experiments/${EVAL_EXP}/best_checkpoint"

echo "Evaluating json file: $OUT_FILE_NAME"

python code/evaluate.py data/tiny-dev.json eval_data/${OUT_FILE_NAME}

