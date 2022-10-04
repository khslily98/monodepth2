# !/bin/bash
MODEL=$1
WEIGHTS=$2

python evaluate_pose.py --data_path KITTI_odom --load_weights_folder checkpoints/$MODEL/models/weights_$WEIGHTS --eval_split odom_9
python evaluate_pose.py --data_path KITTI_odom --load_weights_folder checkpoints/$MODEL/models/weights_$WEIGHTS --eval_split odom_10