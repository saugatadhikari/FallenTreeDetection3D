# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_6000_points_with_interpolation_fg_bg_group_pred_600_trees.py --model="pointnet2_part_seg_msg" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-02-03_14-52_rgbi_4000_pts_overlap_padding_FOCAL_LOSS_FG_BG_ALPHA_None_Gaussian_Wt_True_600_trees_FOLD_1" --num_point=4000 --rgbi_channel --fold_id=1

# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_6000_points_with_interpolation_fg_bg_group_pred_600_trees.py --model="pointnet2_part_seg_msg" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-02-03_22-21_rgbi_4000_pts_overlap_padding_FOCAL_LOSS_FG_BG_ALPHA_None_Gaussian_Wt_True_600_trees_FOLD_2" --num_point=4000 --rgbi_channel --fold_id=2

# # no intensity
# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_6000_points_with_interpolation_fg_bg_group_pred_600_trees_no_intensity.py --model="pointnet2_part_seg_msg" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-02-09_16-24_rgb_4000_pts_overlap_padding_FOCAL_LOSS_FG_BG_ALPHA_None_Gaussian_Wt_True_600_trees_FOLD_0_NO_INTENSITY" --num_point=4000 --rgbi_channel --fold_id=0


# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_6000_points_with_interpolation_fg_bg_group_pred_600_trees.py --model="pointnet2_part_seg_msg" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="post_train_fold_0" --num_point=4000 --rgbi_channel --fold_id=0

# # trained using single block - val using loss
# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_6000_points_with_interpolation_fg_bg_group_pred_600_trees.py --model="pointnet2_part_seg_msg" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-02-14_17-53_rgbi_4000_pts_single_block_FOCAL_LOSS_FG_BG_ALPHA_None_Gaussian_Wt_True_600_trees_FOLD_0" --num_point=4000 --rgbi_channel --fold_id=0

# # trained using single block - val using miou
# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_6000_points_with_interpolation_fg_bg_group_pred_600_trees.py --model="pointnet2_part_seg_msg" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-02-14_17-57_rgbi_4000_pts_single_block_FOCAL_LOSS_FG_BG_ALPHA_None_Gaussian_Wt_True_600_trees_FOLD_0_VAL_MIOU" --num_point=4000 --rgbi_channel --fold_id=0



# # Stage 2

# ALPHA = [2, 0.1]
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_6000_points_with_interpolation_fg_bg_group_pred_stage_2.py --model="pointnet2_part_seg_msg" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-02-18_21-29_rgbi_4000_pts_overlap_padding_FOCAL_LOSS_FG_BG_ALPHA_None_Gaussian_Wt_FALSE_600_trees_STAGE_2_FOLD_0_LATEST
" --num_point=4000 --rgbi_channel

# ALPHA = None
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_6000_points_with_interpolation_fg_bg_group_pred_stage_2.py --model="pointnet2_part_seg_msg" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-02-18_21-33_rgbi_4000_pts_overlap_padding_FOCAL_LOSS_FG_BG_ALPHA_Noneeee_Gaussian_Wt_FALSE_600_trees_STAGE_2_FOLD_0_LATEST_alpha_none" --num_point=4000 --rgbi_channel

# new arch - 4k
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_6000_points_with_interpolation_fg_bg_group_pred_stage_2.py --model="pointnet2_part_seg_msg_4k" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-02-19_14-58_rgbi_4000_pts_overlap_padding_FOCAL_LOSS_FG_BG_ALPHA_Noneeee_Gaussian_Wt_FALSE_600_trees_STAGE_2_FOLD_0_LATEST_alpha_none_4k" --num_point=4000 --rgbi_channel




python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_6000_points_with_interpolation_fg_bg_group_pred_600_trees.py --model="pointnet2_part_seg_msg" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-02-18_10-53_rgbi_4000_pts_overlap_padding_FOCAL_LOSS_FG_BG_ALPHA_None_Gaussian_Wt_True_600_trees_FOLD_1_LATEST" --num_point=4000 --rgbi_channel --fold_id=1

python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_6000_points_with_interpolation_fg_bg_group_pred_600_trees.py --model="pointnet2_part_seg_msg" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-02-18_14-47_rgbi_4000_pts_overlap_padding_FOCAL_LOSS_FG_BG_ALPHA_None_Gaussian_Wt_True_600_trees_FOLD_2_LATEST" --num_point=4000 --rgbi_channel --fold_id=2



#### AFTER NEW WEIGHT ##########

# sigma=1.0, only single FPS blocks
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_6000_points_with_interpolation_fg_bg_group_pred_600_trees.py --model="pointnet2_part_seg_msg" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-02-22_14-24_rgbi_4000_pts_single_block_FOCAL_LOSS_FG_BG_ALPHA_None_Gaussian_Wt_True_600_trees_FOLD_0_SIGMA_1.0" --num_point=4000 --rgbi_channel

# sigma=1.0, only single FPS blocks, subtract wt=1
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_6000_points_with_interpolation_fg_bg_group_pred_600_trees.py --model="pointnet2_part_seg_msg" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-02-22_14-51_rgbi_4000_pts_single_block_FOCAL_LOSS_FG_BG_ALPHA_None_Gaussian_Wt_True_600_trees_FOLD_0_SIGMA_1.0_REPEAT_1_SUB_1" --num_point=4000 --rgbi_channel

# sigma=1.0, only single FPS blocks -> repeat 5 times
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_6000_points_with_interpolation_fg_bg_group_pred_600_trees.py --model="pointnet2_part_seg_msg" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-02-22_14-40_rgbi_4000_pts_single_block_FOCAL_LOSS_FG_BG_ALPHA_None_Gaussian_Wt_True_600_trees_FOLD_0_SIGMA_1.0_REPEAT_5" --num_point=4000 --rgbi_channel


# sigma=1.0, only single FPS blocks -> augment 3 types
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_6000_points_with_interpolation_fg_bg_group_pred_600_trees.py --model="pointnet2_part_seg_msg" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-02-22_15-08_rgbi_4000_pts_single_block_FOCAL_LOSS_FG_BG_ALPHA_None_Gaussian_Wt_True_600_trees_FOLD_0_SIGMA_1.0_REPEAT_0_SUB_0_AUG_3" --num_point=4000 --rgbi_channel


# sigma=1.0, all blocks
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_6000_points_with_interpolation_fg_bg_group_pred_600_trees.py --model="pointnet2_part_seg_msg" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-02-22_16-34_rgbi_4000_pts_ALL_blocks_FOCAL_LOSS_FG_BG_ALPHA_None_Gaussian_Wt_True_600_trees_FOLD_0_SIGMA_1.0" --num_point=4000 --rgbi_channel

# sigma=0.8, all blocks
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_6000_points_with_interpolation_fg_bg_group_pred_600_trees.py --model="pointnet2_part_seg_msg" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-02-22_18-55_rgbi_4000_pts_ALL_blocks_FOCAL_LOSS_FG_BG_ALPHA_None_Gaussian_Wt_True_600_trees_FOLD_0_SIGMA_0.8_crossover_1" --num_point=4000 --rgbi_channel

# fg only wt
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_6000_points_with_interpolation_fg_bg_group_pred_600_trees.py --model="pointnet2_part_seg_msg" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-02-22_19-22_rgbi_4000_pts_ALL_blocks_FOCAL_LOSS_FG_BG_ALPHA_None_Gaussian_Wt_True_600_trees_FOLD_0_SIGMA_1.0_FG_only_wt" --num_point=4000 --rgbi_channel

# sigma = 1.0, alpha=10_1
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_6000_points_with_interpolation_fg_bg_group_pred_600_trees.py --model="pointnet2_part_seg_msg" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-02-22_21-45_rgbi_4000_pts_ALL_blocks_FOCAL_LOSS_FG_BG_Gaussian_Wt_True_600_trees_FOLD_0_SIGMA_1.0_ALPHA_10_1" --num_point=4000 --rgbi_channel










