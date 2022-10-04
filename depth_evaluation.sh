# !/bin/bash
MODEL=$1
WEIGHTS=$2

python evaluate_pose.py --load_weights_folder checkpoints/$MODEL/models/weights_$WEIGHTS --eval_mono
