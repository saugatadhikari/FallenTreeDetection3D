# fold 0, alpha=20_1
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_stage_2.py --epoch=75 --model="pointnet2_part_seg_msg_4k" --tag="FOLD_0_STAGE_2_SYN_DATA" --rgbi_channel --batch_size=4 --npoint=4000 --fold_id=0 --sigma="0.8" --num_aug_copies=3 --alpha_fg=2 --same_weight=0


# fold 1, alpha=20_1
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_stage_2.py --epoch=75 --model="pointnet2_part_seg_msg_4k" --tag="FOLD_1_STAGE_2_SYN_DATA" --rgbi_channel --batch_size=4 --npoint=4000 --fold_id=1 --sigma="0.8" --num_aug_copies=3 --alpha_fg=2 --same_weight=0


# fold 2, alpha=20_1
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_stage_2.py --epoch=75 --model="pointnet2_part_seg_msg_4k" --tag="FOLD_2_STAGE_2_SYN_DATA" --rgbi_channel --batch_size=4 --npoint=4000 --fold_id=2 --sigma="0.8" --num_aug_copies=3 --alpha_fg=2 --same_weight=0


# fold 3, alpha=20_1
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_stage_2.py --epoch=75 --model="pointnet2_part_seg_msg_4k" --tag="FOLD_3_STAGE_2_SYN_DATA" --rgbi_channel --batch_size=4 --npoint=4000 --fold_id=3 --sigma="0.8" --num_aug_copies=3 --alpha_fg=2 --same_weight=0


# fold 4, alpha=20_1
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_stage_2.py --epoch=75 --model="pointnet2_part_seg_msg_4k" --tag="FOLD_4_STAGE_2_SYN_DATA" --rgbi_channel --batch_size=4 --npoint=4000 --fold_id=4 --sigma="0.8" --num_aug_copies=3 --alpha_fg=2 --same_weight=0









# fold 0, alpha=None
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_stage_2.py --epoch=100 --model="pointnet2_part_seg_msg_4k" --tag="STAGE_2_rgb_4k_5_blocks_FG_BG_ALPHA_None_Gaussian_Wt_True_FOLD_0_SIGMA_0.8" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=0 --sigma="0.8" --num_aug_copies=3 --same_weight=0

# fold 0, same weight
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_stage_2.py --epoch=100 --model="pointnet2_part_seg_msg_4k" --tag="STAGE_2_rgb_4k_5_blocks_FG_BG_SAME_WT_FOLD_0" --rgbi_channel --batch_size=4 --npoint=4000 --fold_id=0 --sigma="0.8" --num_aug_copies=3 --same_weight=1

# fold 0, alpha=5_1
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_stage_2.py --epoch=100 --model="pointnet2_part_seg_msg_4k" --tag="STAGE_2_rgb_4k_5_blocks_FG_BG_ALPHA_5_1_Gaussian_Wt_True_FOLD_0_SIGMA_0.8" --rgbi_channel --batch_size=4 --npoint=4000 --fold_id=0 --sigma="0.8" --num_aug_copies=3 --alpha_fg="0.5" --same_weight=0


# fold 0, alpha=2_1
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_stage_2.py --epoch=100 --model="pointnet2_part_seg_msg_4k" --tag="STAGE_2_rgb_4k_5_blocks_FG_BG_ALPHA_2_1_Gaussian_Wt_True_FOLD_0_SIGMA_0.8" --rgbi_channel --batch_size=4 --npoint=4000 --fold_id=0 --sigma="0.8" --num_aug_copies=3 --alpha_fg="0.2" --same_weight=0


# fold 0, alpha=3_1
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_stage_2.py --epoch=100 --model="pointnet2_part_seg_msg_4k" --tag="STAGE_2_rgb_4k_5_blocks_FG_BG_ALPHA_3_1_Gaussian_Wt_True_FOLD_0_SIGMA_0.8" --rgbi_channel --batch_size=4 --npoint=4000 --fold_id=0 --sigma="0.8" --num_aug_copies=3 --alpha_fg="0.3" --same_weight=0


# fold 0, alpha=4_1
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_stage_2.py --epoch=100 --model="pointnet2_part_seg_msg_4k" --tag="STAGE_2_rgb_4k_5_blocks_FG_BG_ALPHA_4_1_Gaussian_Wt_True_FOLD_0_SIGMA_0.8" --rgbi_channel --batch_size=4 --npoint=4000 --fold_id=0 --sigma="0.8" --num_aug_copies=3 --alpha_fg="0.4" --same_weight=0


# fold 0, alpha=20_1, new loss (demote False Positives), alpha_false_positives=1
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_stage_2.py --epoch=100 --model="pointnet2_part_seg_msg_4k" --tag="STAGE_2_rgb_4k_5_blocks_FG_BG_ALPHA_20_1_Gaussian_Wt_True_FOLD_0_SIGMA_0.8_MORE_DATA_NEW_LOSS" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=0 --sigma="0.8" --num_aug_copies=3 --alpha_fg=2 --same_weight=0

# fold 0, alpha=20_1, new loss (demote False Positives), alpha_false_positives=1, continue from stage 1 model
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_stage_2.py --epoch=200 --model="pointnet2_part_seg_msg_4k" --tag="STAGE_2_rgb_4k_5_blocks_FG_BG_ALPHA_20_1_Gaussian_Wt_True_FOLD_0_SIGMA_0.8_MORE_DATA_NEW_LOSS_POST_TRAIN" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=0 --sigma="0.8" --num_aug_copies=3 --alpha_fg=2 --same_weight=0 --log_dir="2026-03-03_23-13_rgb_4k_5_blocks_FG_BG_ALPHA_20_1_Gaussian_Wt_True_FOLD_0_SIGMA_0.8_CROSSOVER_ALPHA"


# fold 1, alpha=20_1, alpha_false_positives=1
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_stage_2_hard_neg_none.py --epoch=200 --model="pointnet2_part_seg_msg_4k" --tag="STAGE_2_rgb_4k_5_blocks_FG_BG_ALPHA_20_1_Gaussian_Wt_True_FOLD_1_SIGMA_0.8_MORE_DATA_NO_FP_DEMOTE" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=1 --sigma="0.8" --num_aug_copies=3 --alpha_fg=2 --same_weight=0


# fold 1, alpha=20_1, new loss (demote False Positives), alpha_false_positives=1
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_stage_2.py --epoch=100 --model="pointnet2_part_seg_msg_4k" --tag="STAGE_2_rgb_4k_5_blocks_FG_BG_ALPHA_20_1_Gaussian_Wt_True_FOLD_1_SIGMA_0.8_MORE_DATA_NEW_LOSS" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=1 --sigma="0.8" --num_aug_copies=3 --alpha_fg=2 --same_weight=0

# fold 1, alpha=None, alpha_false_positives=1
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_stage_2_hard_neg_none.py --epoch=100 --model="pointnet2_part_seg_msg_4k" --tag="STAGE_2_rgb_4k_5_blocks_FG_BG_ALPHA_None_Gaussian_Wt_True_FOLD_1_SIGMA_0.8_MORE_DATA_NO_FP_DEMOTE" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=1 --sigma="0.8" --num_aug_copies=3 --same_weight=0

# fold 1, alpha=5_1
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_stage_2_hard_neg_none.py --epoch=100 --model="pointnet2_part_seg_msg_4k" --tag="STAGE_2_rgb_4k_5_blocks_FG_BG_ALPHA_5_1_Gaussian_Wt_True_FOLD_1_SIGMA_0.8" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=1 --sigma="0.8" --num_aug_copies=3 --alpha_fg="0.5" --same_weight=0

# fold 1, alpha=10_1
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_stage_2_hard_neg_none.py --epoch=100 --model="pointnet2_part_seg_msg_4k" --tag="STAGE_2_rgb_4k_5_blocks_FG_BG_ALPHA_10_1_Gaussian_Wt_True_FOLD_1_SIGMA_0.8" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=1 --sigma="0.8" --num_aug_copies=3 --alpha_fg="1" --same_weight=0

# fold 1, alpha=3_1
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_stage_2_hard_neg_none.py --epoch=100 --model="pointnet2_part_seg_msg_4k" --tag="STAGE_2_rgb_4k_5_blocks_FG_BG_ALPHA_3_1_Gaussian_Wt_True_FOLD_1_SIGMA_0.8" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=1 --sigma="0.8" --num_aug_copies=3 --alpha_fg="0.3" --same_weight=0



# fold 1, alpha=20_1, small radius
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_stage_2_hard_neg_none.py --epoch=200 --model="pointnet2_part_seg_msg_4k_small_radius" --tag="STAGE_2_rgb_4k_5_blocks_FG_BG_ALPHA_20_1_Gaussian_Wt_True_FOLD_1_SIGMA_0.8_MORE_DATA_NO_FP_DEMOTE_SMALL_RADIUS" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=1 --sigma="0.8" --num_aug_copies=3 --alpha_fg=2 --same_weight=0


# fold 1, alpha=None, small radius
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_stage_2_hard_neg_none.py --epoch=200 --model="pointnet2_part_seg_msg_4k_small_radius" --tag="STAGE_2_rgb_4k_5_blocks_FG_BG_ALPHA_None_Gaussian_Wt_True_FOLD_1_SIGMA_0.8_MORE_DATA_NO_FP_DEMOTE_SMALL_RADIUS" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=1 --sigma="0.8" --num_aug_copies=3 --same_weight=0


# fold 2, alpha=20_1
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_stage_2_hard_neg_none.py --epoch=100 --model="pointnet2_part_seg_msg_4k" --tag="STAGE_2_rgb_4k_5_blocks_FG_BG_ALPHA_20_1_Gaussian_Wt_True_FOLD_2_SIGMA_0.8_MORE_DATA_NO_FP_DEMOTE" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=2 --sigma="0.8" --num_aug_copies=3 --alpha_fg=2 --same_weight=0





# fold 0, alpha=20_1
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_stage_2_hard_neg_none.py --epoch=100 --model="pointnet2_part_seg_msg_4k" --tag="STAGE_2_FOLD_0_CW_FACTOR" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=0 --sigma="0.8" --num_aug_copies=3 --alpha_fg=2 --same_weight=0

# fold 1, alpha=20_1
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_stage_2_hard_neg_none.py --epoch=100 --model="pointnet2_part_seg_msg_4k" --tag="STAGE_2_FOLD_1_CW_FACTOR" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=1 --sigma="0.8" --num_aug_copies=3 --alpha_fg=2 --same_weight=0

# fold 2, alpha=20_1
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_stage_2_hard_neg_none.py --epoch=100 --model="pointnet2_part_seg_msg_4k" --tag="STAGE_2_FOLD_2_CW_FACTOR" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=2 --sigma="0.8" --num_aug_copies=3 --alpha_fg=2 --same_weight=0

# fold 3, alpha=20_1
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_stage_2_hard_neg_none.py --epoch=100 --model="pointnet2_part_seg_msg_4k" --tag="STAGE_2_FOLD_3_CW_FACTOR" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=3 --sigma="0.8" --num_aug_copies=3 --alpha_fg=2 --same_weight=0

# fold 4, alpha=20_1
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_stage_2_hard_neg_none.py --epoch=100 --model="pointnet2_part_seg_msg_4k" --tag="STAGE_2_FOLD_4_CW_FACTOR" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=4 --sigma="0.8" --num_aug_copies=3 --alpha_fg=2 --same_weight=0
