#!/bin/sh
srun python itr_tools/df_to_json.py --dataset fashion200k --dataset_split train
srun python itr_tools/df_to_json.py --dataset fashion200k --dataset_split test
srun python itr_tools/df_to_json.py --dataset fashion200k --dataset_split dev