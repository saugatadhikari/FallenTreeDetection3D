# # Train Stage 1
# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/post_train_stage_1_SYNTHETIC_DATA.py --epoch=200 --model="pointnet2_part_seg_msg_4k" --tag="FOLD_0_SYN_POST_TRAIN" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=0 --sigma="0.8" --num_aug_copies=3 --alpha_fg="2" --pretrained_model="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/tree_log/pointnet2_part_seg_msg_4k/BEST_MODELS_FOR_2D_STAGE_1/checkpoints/best_model.pth"


# # Train Stage 2
# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/post_train_stage_2_SYNTHETIC_DATA.py --epoch=100 --model="pointnet2_part_seg_msg_4k" --tag="FOLD_0_SYN_POST_TRAIN_STAGE_2" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=0 --sigma="0.8" --num_aug_copies=3 --alpha_fg="2" --pretrained_model="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/tree_log/pointnet2_part_seg_msg_4k/BEST_MODELS_FOR_2D_STAGE_2/checkpoints/best_model.pth"


# # Train Stage 3
# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/post_train_stage_3_SYNTHETIC_DATA.py --epoch=130 --model="pointnet2_part_seg_msg_4k_stage_3" --tag="FOLD_0_SYN_POST_TRAIN_STAGE_3" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=0 --sigma="0.8" --num_aug_copies=3 --alpha_fg="2" --pretrained_model="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/tree_log/pointnet2_part_seg_msg_4k_stage_3/BEST_MODELS_FOR_2D_STAGE_3/checkpoints/best_model.pth" --same_weight=1



# Using all data (original and synthetic) - from scratch

# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/post_train_stage_1_SYNTHETIC_DATA_PLUS_ORIG_DATA.py --epoch=50 --model="pointnet2_part_seg_msg_4k" --tag="FOLD_0_SYN_PLUS_ORIG" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=0 --sigma="0.8" --num_aug_copies=3 --alpha_fg="2"


python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/post_train_stage_1_SYNTHETIC_DATA_PLUS_ORIG_DATA.py --epoch=25 --model="pointnet2_part_seg_msg_4k" --tag="FOLD_1_SYN_PLUS_ORIG" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=1 --sigma="0.8" --num_aug_copies=3 --alpha_fg="2"

python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/post_train_stage_1_SYNTHETIC_DATA_PLUS_ORIG_DATA.py --epoch=25 --model="pointnet2_part_seg_msg_4k" --tag="FOLD_2_SYN_PLUS_ORIG" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=2 --sigma="0.8" --num_aug_copies=3 --alpha_fg="2"


python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/post_train_stage_1_SYNTHETIC_DATA_PLUS_ORIG_DATA.py --epoch=20 --model="pointnet2_part_seg_msg_4k" --tag="FOLD_2_SYN_PLUS_ORIG" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=2 --sigma="0.8" --num_aug_copies=3 --alpha_fg="2" --pretrained_model="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/scripts/tree_log/pointnet2_part_seg_msg_4k/2026-07-08_06-28_FOLD_2_SYN_PLUS_ORIG/checkpoints/best_model.pth"

python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/post_train_stage_1_SYNTHETIC_DATA_PLUS_ORIG_DATA.py --epoch=20 --model="pointnet2_part_seg_msg_4k" --tag="FOLD_4_SYN_PLUS_ORIG" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=4 --sigma="0.8" --num_aug_copies=3 --alpha_fg="2"


python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/post_train_stage_1_SYNTHETIC_DATA_PLUS_ORIG_DATA.py --epoch=30 --model="pointnet2_part_seg_msg_4k" --tag="FOLD_3_SYN_PLUS_ORIG" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=3 --sigma="0.8" --num_aug_copies=3 --alpha_fg="2" --pretrained_model="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/scripts/tree_log/pointnet2_part_seg_msg_4k/2026-07-08_14-14_FOLD_3_SYN_PLUS_ORIG/checkpoints/best_model.pth" # just continuing from last time, not a model from months earlier (w/o synthetic data)


# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/post_train_stage_1_SYNTHETIC_DATA_PLUS_ORIG_DATA.py --epoch=20 --model="pointnet2_part_seg_msg_4k" --tag="FOLD_3_SYN_PLUS_ORIG" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=3 --sigma="0.8" --num_aug_copies=3 --alpha_fg="2"
