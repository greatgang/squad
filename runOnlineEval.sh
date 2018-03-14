#!/bin/bash

echo "Running official evaluation:"

cl run --name gen-answers --request-docker-image abisee/cs224n-dfp:v4 :code :best_checkpoint glove.txt:0x97c870/glove.6B.100d.txt data.json:0x4870af 'python code/main.py --mode=official_eval --glove_path=glove.txt --json_in_path=data.json --ckpt_load_dir=best_checkpoint' --request-cpus 1 --request-memory 4g --request-disk 1g --request-time 1d

