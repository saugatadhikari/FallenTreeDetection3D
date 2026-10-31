# # fold 0, alpha=20_1
# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_6000_points_with_interpolation_fg_bg_group_pred_stage_2.py --model="pointnet2_part_seg_msg_4k" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-03-06_23-30_STAGE_2_rgb_4k_5_blocks_FG_BG_ALPHA_20_1_Gaussian_Wt_True_FOLD_0_SIGMA_0.8" --num_point=4000 --rgbi_channel --fold_id=0

# # fold 0, alpha=None
# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_6000_points_with_interpolation_fg_bg_group_pred_stage_2.py --model="pointnet2_part_seg_msg_4k" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-03-06_23-17_STAGE_2_rgb_4k_5_blocks_FG_BG_ALPHA_None_Gaussian_Wt_True_FOLD_0_SIGMA_0.8" --num_point=4000 --rgbi_channel --fold_id=0

# # fold 0, same weight
# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_6000_points_with_interpolation_fg_bg_group_pred_stage_2.py --model="pointnet2_part_seg_msg_4k" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-03-06_23-23_STAGE_2_rgb_4k_5_blocks_FG_BG_SAME_WT_FOLD_0" --num_point=4000 --rgbi_channel --fold_id=0

# # fold 0, alpha=5_1
# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_6000_points_with_interpolation_fg_bg_group_pred_stage_2.py --model="pointnet2_part_seg_msg_4k" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-03-07_08-23_STAGE_2_rgb_4k_5_blocks_FG_BG_ALPHA_5_1_Gaussian_Wt_True_FOLD_0_SIGMA_0.8" --num_point=4000 --rgbi_channel --fold_id=0

# # fold 0, alpha=2_1
# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_6000_points_with_interpolation_fg_bg_group_pred_stage_2.py --model="pointnet2_part_seg_msg_4k" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-03-08_19-27_STAGE_2_rgb_4k_5_blocks_FG_BG_ALPHA_2_1_Gaussian_Wt_True_FOLD_0_SIGMA_0.8" --num_point=4000 --rgbi_channel --fold_id=0

# # fold 0, alpha=3_1
# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_6000_points_with_interpolation_fg_bg_group_pred_stage_2.py --model="pointnet2_part_seg_msg_4k" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-03-08_19-28_STAGE_2_rgb_4k_5_blocks_FG_BG_ALPHA_3_1_Gaussian_Wt_True_FOLD_0_SIGMA_0.8" --num_point=4000 --rgbi_channel --fold_id=0


# # fold 0, alpha=20_1, new loss (demote False Positives), alpha_false_positives=1
# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_6000_points_with_interpolation_fg_bg_group_pred_stage_2.py --model="pointnet2_part_seg_msg_4k" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-03-09_17-06_STAGE_2_rgb_4k_5_blocks_FG_BG_ALPHA_20_1_Gaussian_Wt_True_FOLD_0_SIGMA_0.8_MORE_DATA_NEW_LOSS" --num_point=4000 --rgbi_channel --fold_id=0


# # fold 0, alpha=20_1, new loss (demote False Positives), alpha_false_positives=1, continue from stage 1 model
# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_6000_points_with_interpolation_fg_bg_group_pred_stage_2.py --model="pointnet2_part_seg_msg_4k" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-03-03_23-13_rgb_4k_5_blocks_FG_BG_ALPHA_20_1_Gaussian_Wt_True_FOLD_0_SIGMA_0.8_CROSSOVER_ALPHA_STAGE_2_rgb_4k_5_blocks_FG_BG_ALPHA_20_1_Gaussian_Wt_True_FOLD_0_SIGMA_0.8_MORE_DATA_NEW_LOSS_POST_TRAIN" --num_point=4000 --rgbi_channel --fold_id=0


# # fold 1, alpha=20_1, new loss (demote False Positives), alpha_false_positives=1
# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_6000_points_with_interpolation_fg_bg_group_pred_stage_2.py --model="pointnet2_part_seg_msg_4k" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-03-18_19-01_STAGE_2_rgb_4k_5_blocks_FG_BG_ALPHA_20_1_Gaussian_Wt_True_FOLD_1_SIGMA_0.8_MORE_DATA_NEW_LOSS" --num_point=4000 --rgbi_channel --fold_id=1





# # fold 1, alpha=None, NO DEMOTE
# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_6000_points_with_interpolation_fg_bg_group_pred_stage_2.py --model="pointnet2_part_seg_msg_4k" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-03-19_12-31_STAGE_2_rgb_4k_5_blocks_FG_BG_ALPHA_None_Gaussian_Wt_True_FOLD_1_SIGMA_0.8_MORE_DATA_NO_FP_DEMOTE" --num_point=4000 --rgbi_channel --fold_id=1

# # fold 1, alpha=20_1, DEMOTE FP
# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_6000_points_with_interpolation_fg_bg_group_pred_stage_2.py --model="pointnet2_part_seg_msg_4k" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-03-19_11-10_STAGE_2_rgb_4k_5_blocks_FG_BG_ALPHA_20_1_Gaussian_Wt_True_FOLD_1_SIGMA_0.8_MORE_DATA_NEW_LOSS" --num_point=4000 --rgbi_channel --fold_id=1

# # fold 1, alpha=20_1, NO DEMOTE FP
# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_6000_points_with_interpolation_fg_bg_group_pred_stage_2.py --model="pointnet2_part_seg_msg_4k" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-03-19_14-44_STAGE_2_rgb_4k_5_blocks_FG_BG_ALPHA_20_1_Gaussian_Wt_True_FOLD_1_SIGMA_0.8_MORE_DATA_NO_FP_DEMOTE" --num_point=4000 --rgbi_channel --fold_id=1




# # fold 1, alpha=5_1, NO DEMOTE FP
# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_6000_points_with_interpolation_fg_bg_group_pred_stage_2.py --model="pointnet2_part_seg_msg_4k" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-03-20_08-51_STAGE_2_rgb_4k_5_blocks_FG_BG_ALPHA_5_1_Gaussian_Wt_True_FOLD_1_SIGMA_0.8" --num_point=4000 --rgbi_channel --fold_id=1


# # fold 2, alpha=10_1, NO DEMOTE FP
# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_6000_points_with_interpolation_fg_bg_group_pred_stage_2.py --model="pointnet2_part_seg_msg_4k" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-03-23_11-35_STAGE_2_rgb_4k_5_blocks_FG_BG_ALPHA_20_1_Gaussian_Wt_True_FOLD_2_SIGMA_0.8_MORE_DATA_NO_FP_DEMOTE" --num_point=4000 --rgbi_channel --fold_id=2

# # fold 3, alpha=10_1, NO DEMOTE FP
# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_6000_points_with_interpolation_fg_bg_group_pred_stage_2.py --model="pointnet2_part_seg_msg_4k" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="STAGE_2_FOLD_3_INFER" --num_point=4000 --rgbi_channel --fold_id=3

# # fold 4, alpha=10_1, NO DEMOTE FP
# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_6000_points_with_interpolation_fg_bg_group_pred_stage_2.py --model="pointnet2_part_seg_msg_4k" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="STAGE_2_FOLD_4_INFER" --num_point=4000 --rgbi_channel --fold_id=4





# # fold 0, alpha=10_1
# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_6000_points_with_interpolation_fg_bg_group_pred_stage_2.py --model="pointnet2_part_seg_msg_4k" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-03-24_17-50_STAGE_2_FOLD_0_CW" --num_point=4000 --rgbi_channel --fold_id=0


# # fold 1, alpha=10_1
# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_6000_points_with_interpolation_fg_bg_group_pred_stage_2.py --model="pointnet2_part_seg_msg_4k" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-03-24_17-51_STAGE_2_FOLD_1_CW" --num_point=4000 --rgbi_channel --fold_id=1


# # fold 2, alpha=10_1
# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_6000_points_with_interpolation_fg_bg_group_pred_stage_2.py --model="pointnet2_part_seg_msg_4k" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-03-24_17-52_STAGE_2_FOLD_2_CW" --num_point=4000 --rgbi_channel --fold_id=2


# # fold 3, alpha=10_1
# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_6000_points_with_interpolation_fg_bg_group_pred_stage_2.py --model="pointnet2_part_seg_msg_4k" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-03-24_17-52_STAGE_2_FOLD_3_CW" --num_point=4000 --rgbi_channel --fold_id=3

# # fold 4, alpha=10_1
# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_6000_points_with_interpolation_fg_bg_group_pred_stage_2.py --model="pointnet2_part_seg_msg_4k" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-03-24_17-53_STAGE_2_FOLD_4_CW" --num_point=4000 --rgbi_channel --fold_id=4


# # fold 0, alpha=10_1
# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_6000_points_with_interpolation_fg_bg_group_pred_stage_2.py --model="pointnet2_part_seg_msg_4k" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-03-25_09-42_STAGE_2_FOLD_0_CW_FACTOR" --num_point=4000 --rgbi_channel --fold_id=0

# # fold 1, alpha=10_1
# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_6000_points_with_interpolation_fg_bg_group_pred_stage_2.py --model="pointnet2_part_seg_msg_4k" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-03-25_09-43_STAGE_2_FOLD_1_CW_FACTOR" --num_point=4000 --rgbi_channel --fold_id=1

# # fold 2, alpha=10_1
# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_6000_points_with_interpolation_fg_bg_group_pred_stage_2.py --model="pointnet2_part_seg_msg_4k" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-03-25_10-51_STAGE_2_FOLD_2_CW_FACTOR" --num_point=4000 --rgbi_channel --fold_id=2

# fold 3, alpha=10_1
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_6000_points_with_interpolation_fg_bg_group_pred_stage_2.py --model="pointnet2_part_seg_msg_4k" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-03-25_20-33_STAGE_2_FOLD_3_CW_FACTOR" --num_point=4000 --rgbi_channel --fold_id=3

# fold 4, alpha=10_1
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_6000_points_with_interpolation_fg_bg_group_pred_stage_2.py --model="pointnet2_part_seg_msg_4k" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-03-25_20-46_STAGE_2_FOLD_4_CW_FACTOR" --num_point=4000 --rgbi_channel --fold_id=4


