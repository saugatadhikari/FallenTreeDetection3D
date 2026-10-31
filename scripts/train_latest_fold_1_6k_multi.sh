# # Fold 1; train larger model; Sigma 0.8, alpha=None, intensity=0, augment=3
# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_600_trees_ignore_handled_no_intensity.py --epoch=100 --model="pointnet2_part_seg_msg_4k" --tag="rgb_6k_5_blocks_FG_BG_ALPHA_None_Gaussian_Wt_True_FOLD_1_SIGMA_0.8" --rgbi_channel --batch_size=4 --npoint=6000 --fold_id=1 --sigma="0.8" --num_aug_copies=3

# Fold 1; train larger model; Sigma 1.0, alpha=None, intensity=0, augment=3
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_600_trees_ignore_handled_no_intensity.py --epoch=100 --model="pointnet2_part_seg_msg_4k" --tag="rgb_6k_5_blocks_FG_BG_ALPHA_None_Gaussian_Wt_True_FOLD_1_SIGMA_1.0" --rgbi_channel --batch_size=4 --npoint=6000 --fold_id=1 --sigma="1.0" --num_aug_copies=3

# Fold 1; train larger model; Sigma 1.2, alpha=None, intensity=0, augment=3
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_600_trees_ignore_handled_no_intensity.py --epoch=100 --model="pointnet2_part_seg_msg_4k" --tag="rgb_6k_5_blocks_FG_BG_ALPHA_None_Gaussian_Wt_True_FOLD_1_SIGMA_1.2" --rgbi_channel --batch_size=4 --npoint=6000 --fold_id=1 --sigma="1.2" --num_aug_copies=3


# Fold 1; train larger model; Sigma 1.2, alpha=None, intensity=0, augment=3
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_600_trees_ignore_handled_no_intensity.py --epoch=100 --model="pointnet2_part_seg_msg_4k" --tag="rgb_4k_5_blocks_FG_BG_ALPHA_None_Gaussian_Wt_True_FOLD_1_SIGMA_1.2" --rgbi_channel --batch_size=4 --npoint=4000 --fold_id=1 --sigma="1.2" --num_aug_copies=3

python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_600_trees_ignore_handled_no_intensity.py --epoch=100 --model="pointnet2_part_seg_msg_4k" --tag="rgb_4k_5_blocks_FG_BG_ALPHA_None_Gaussian_Wt_True_FOLD_1_SIGMA_0.8_NEW" --rgbi_channel --batch_size=4 --npoint=4000 --fold_id=1 --sigma="0.8" --num_aug_copies=3

# Fold 1; train larger model; Sigma 1.0, alpha=None, intensity=0, augment=3
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_600_trees_ignore_handled.py --epoch=100 --model="pointnet2_part_seg_msg_4k" --tag="rgb_4k_5_blocks_FG_BG_ALPHA_None_Gaussian_Wt_True_FOLD_1_SIGMA_0.8_INTENSITY_TRUE" --rgbi_channel --batch_size=4 --npoint=4000 --fold_id=1 --sigma="0.8" --num_aug_copies=3


# training on tmux0...
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_600_trees_ignore_handled_no_intensity_min_wt_0.py --epoch=100 --model="pointnet2_part_seg_msg_4k" --tag="rgb_4k_5_blocks_FG_BG_ALPHA_None_Gaussian_Wt_True_FOLD_1_SIGMA_0.8_MIN_WT_0" --rgbi_channel --batch_size=4 --npoint=4000 --fold_id=1 --sigma="0.8" --num_aug_copies=3


# sigma dynamic
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_600_trees_ignore_handled_no_intensity.py --epoch=100 --model="pointnet2_part_seg_msg_4k" --tag="rgb_4k_5_blocks_FG_BG_ALPHA_None_Gaussian_Wt_True_FOLD_1_SIGMA_dynamic" --rgbi_channel --batch_size=4 --npoint=4000 --fold_id=1 --sigma="100" --num_aug_copies=3

# alpha = 1, 0.1
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_600_trees_ignore_handled.py --epoch=100 --model="pointnet2_part_seg_msg_4k" --tag="rgb_4k_5_blocks_FG_BG_ALPHA_10_1_Gaussian_Wt_True_FOLD_1_SIGMA_0.8" --rgbi_channel --batch_size=4 --npoint=4000 --fold_id=1 --sigma="0.8" --num_aug_copies=3 --alpha_fg=1

# alpha = 2, 0.1
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_600_trees_ignore_handled.py --epoch=100 --model="pointnet2_part_seg_msg_4k" --tag="rgb_4k_5_blocks_FG_BG_ALPHA_20_1_Gaussian_Wt_True_FOLD_1_SIGMA_0.8" --rgbi_channel --batch_size=4 --npoint=4000 --fold_id=1 --sigma="0.8" --num_aug_copies=3 --alpha_fg=2

# alpha = 3, 0.1
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_600_trees_ignore_handled.py --epoch=100 --model="pointnet2_part_seg_msg_4k" --tag="rgb_4k_5_blocks_FG_BG_ALPHA_30_1_Gaussian_Wt_True_FOLD_1_SIGMA_0.8" --rgbi_channel --batch_size=4 --npoint=4000 --fold_id=1 --sigma="0.8" --num_aug_copies=3 --alpha_fg=3



# sigma dynamic, alpha = 1, 0.1
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_600_trees_ignore_handled.py --epoch=100 --model="pointnet2_part_seg_msg_4k" --tag="rgb_4k_5_blocks_FG_BG_ALPHA_10_1_Gaussian_Wt_True_FOLD_1_SIGMA_dynamic" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=1 --sigma="100" --num_aug_copies=3 --alpha_fg=1

# sigma dynamic, alpha = 2, 0.1
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_600_trees_ignore_handled.py --epoch=100 --model="pointnet2_part_seg_msg_4k" --tag="rgb_4k_5_blocks_FG_BG_ALPHA_20_1_Gaussian_Wt_True_FOLD_1_SIGMA_dynamic" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=1 --sigma="100" --num_aug_copies=3 --alpha_fg=2

# sigma dynamic, alpha = 3, 0.1
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_600_trees_ignore_handled.py --epoch=100 --model="pointnet2_part_seg_msg_4k" --tag="rgb_4k_5_blocks_FG_BG_ALPHA_30_1_Gaussian_Wt_True_FOLD_1_SIGMA_dynamic" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=1 --sigma="100" --num_aug_copies=3 --alpha_fg=3



# sigma 0.8, 1 block, alpha=None
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_600_trees_ignore_handled_single_block.py --epoch=100 --model="pointnet2_part_seg_msg_4k" --tag="rgb_4k_1_blocks_FG_BG_ALPHA_None_Gaussian_Wt_True_FOLD_1_SIGMA_0.8" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=1 --sigma="0.8" --num_aug_copies=3 --alpha_fg=1

# sigma 0.8, 1 block, alpha=20_1, AUG=0
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_600_trees_ignore_handled_single_block.py --epoch=300 --model="pointnet2_part_seg_msg_4k" --tag="rgb_4k_1_blocks_FG_BG_ALPHA_20_1_Gaussian_Wt_True_FOLD_1_SIGMA_0.8_AUG_0" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=1 --sigma="0.8" --num_aug_copies=0 --alpha_fg=2 --log_dir="2026-03-03_21-35_rgb_4k_1_blocks_FG_BG_ALPHA_20_1_Gaussian_Wt_True_FOLD_1_SIGMA_0.8_AUG_0"

# sigma 0.8, 1 block, alpha=20_1, AUG=3
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_600_trees_ignore_handled_single_block.py --epoch=100 --model="pointnet2_part_seg_msg_4k" --tag="rgb_4k_1_blocks_FG_BG_ALPHA_20_1_Gaussian_Wt_True_FOLD_1_SIGMA_0.8_AUG_3" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=1 --sigma="0.8" --num_aug_copies=3 --alpha_fg=2


# alpha = 2, 0.1
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_600_trees_ignore_handled.py --epoch=100 --model="pointnet2_part_seg_msg_4k" --tag="rgb_4k_5_blocks_FG_BG_ALPHA_20_1_Gaussian_Wt_True_FOLD_1_SIGMA_0.8_CROSSOVER_ALPHA" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=1 --sigma="0.8" --num_aug_copies=3 --alpha_fg=2


# alpha = 2, 0.1, Fold 0
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_600_trees_ignore_handled.py --epoch=100 --model="pointnet2_part_seg_msg_4k" --tag="rgb_4k_5_blocks_FG_BG_ALPHA_20_1_Gaussian_Wt_True_FOLD_0_SIGMA_0.8" --rgbi_channel --batch_size=4 --npoint=4000 --fold_id=0 --sigma="0.8" --num_aug_copies=3 --alpha_fg=2

# alpha = 2, 0.1, Fold 2
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_600_trees_ignore_handled.py --epoch=100 --model="pointnet2_part_seg_msg_4k" --tag="rgb_4k_5_blocks_FG_BG_ALPHA_20_1_Gaussian_Wt_True_FOLD_2_SIGMA_0.8" --rgbi_channel --batch_size=4 --npoint=4000 --fold_id=2 --sigma="0.8" --num_aug_copies=3 --alpha_fg=2

# alpha = 2, 0.1, Fold 3
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_600_trees_ignore_handled.py --epoch=100 --model="pointnet2_part_seg_msg_4k" --tag="rgb_4k_5_blocks_FG_BG_ALPHA_20_1_Gaussian_Wt_True_FOLD_3_SIGMA_0.8" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=3 --sigma="0.8" --num_aug_copies=3 --alpha_fg=2

# alpha = 2, 0.1, Fold 4
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_600_trees_ignore_handled.py --epoch=100 --model="pointnet2_part_seg_msg_4k" --tag="rgb_4k_5_blocks_FG_BG_ALPHA_20_1_Gaussian_Wt_True_FOLD_4_SIGMA_0.8" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=4 --sigma="0.8" --num_aug_copies=3 --alpha_fg=2



# 4k, intensity, 0.8, small model, alpha=None
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_600_trees_ignore_handled.py --epoch=100 --model="pointnet2_part_seg_msg" --tag="rgb_4k_5_blocks_FG_BG_ALPHA_None_Gaussian_Wt_True_FOLD_1_SIGMA_0.8" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=1 --sigma="0.8" --num_aug_copies=3

# 6k, intensity, 0.8, small model, alpha=None
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_600_trees_ignore_handled.py --epoch=100 --model="pointnet2_part_seg_msg" --tag="rgb_6k_5_blocks_FG_BG_ALPHA_None_Gaussian_Wt_True_FOLD_1_SIGMA_0.8" --rgbi_channel --batch_size=8 --npoint=6000 --fold_id=1 --sigma="0.8" --num_aug_copies=3


# alpha = 2, 0.1; small model - 4k
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_600_trees_ignore_handled.py --epoch=100 --model="pointnet2_part_seg_msg" --tag="rgb_4k_5_blocks_FG_BG_ALPHA_20_1_Gaussian_Wt_True_FOLD_1_SIGMA_0.8_small_model" --rgbi_channel --batch_size=4 --npoint=4000 --fold_id=1 --sigma="0.8" --num_aug_copies=3 --alpha_fg=2


# alpha = 2, 0.1; small model - 6k
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_600_trees_ignore_handled.py --epoch=100 --model="pointnet2_part_seg_msg" --tag="rgb_6k_5_blocks_FG_BG_ALPHA_20_1_Gaussian_Wt_True_FOLD_1_SIGMA_0.8_small_model" --rgbi_channel --batch_size=4 --npoint=6000 --fold_id=1 --sigma="0.8" --num_aug_copies=3 --alpha_fg=2

# 4k large model, wt based on factor
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_600_trees_ignore_handled.py --epoch=100 --model="pointnet2_part_seg_msg_4k" --tag="rgb_4k_5_blocks_FG_BG_ALPHA_factor_1_Gaussian_Wt_True_FOLD_1_SIGMA_0.8" --rgbi_channel --batch_size=4 --npoint=4000 --fold_id=1 --sigma="0.8" --num_aug_copies=3 --alpha_factor=1

