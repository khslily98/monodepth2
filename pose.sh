# !/bin/bash
MODEL=$1
WEIGHTS=$2

echo "Pose eval start!! : $0 "

python evaluate_pose.py --data_path /mnt/server8_hard0/shkim/KITTI_odom --load_weights_folder /mnt/server4_hard1/heeseon/monodepth2/checkpoints/$MODEL/models/$WEIGHTS --model_name $MODEL --weight_name $WEIGHTS 

# echo "$MODEL, $WEIGHTS"
# echo "model name : $MODEL"
# echo "weights : $WEIGHTS "
# print_usage()
# {
#   echo "Usage: $0 filename lines"
#   exit 2
# }

# if [[ -d ~/datasets/checkpoints/$MODEL ]] 
# then 
#     python evaluate_pose.py --data_path ~/datasets/KITTI_odom --png --load_weights_folder ~/datasets/checkpoints/$MODEL/models/$WEIGHTS --eval_split odom_0
#     python evaluate_pose.py --data_path ~/datasets/KITTI_odom --png --load_weights_folder ~/datasets/checkpoints/$MODEL/models/$WEIGHTS --eval_split odom_3
#     python evaluate_pose.py --data_path ~/datasets/KITTI_odom --png --load_weights_folder ~/datasets/checkpoints/$MODEL/models/$WEIGHTS --eval_split odom_4
#     python evaluate_pose.py --data_path ~/datasets/KITTI_odom --png --load_weights_folder ~/datasets/checkpoints/$MODEL/models/$WEIGHTS --eval_split odom_5
#     python evaluate_pose.py --data_path ~/datasets/KITTI_odom --png --load_weights_folder ~/datasets/checkpoints/$MODEL/models/$WEIGHTS --eval_split odom_7
#     python evaluate_pose.py --data_path ~/datasets/KITTI_odom --png --load_weights_folder ~/datasets/checkpoints/$MODEL/models/$WEIGHTS --eval_split odom_9
#     python evaluate_pose.py --data_path ~/datasets/KITTI_odom --png --load_weights_folder ~/datasets/checkpoints/$MODEL/models/$WEIGHTS --eval_split odom_10
# fi