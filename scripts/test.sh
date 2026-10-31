# XYZ only - 120 standing, 60 fallen
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_treepartseg_all_pts.py --model="pointnet_part_seg" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2025-07-30_12-12_xyz_new_sampling"

# XYZ only - 120 standing, 60 fallen
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_treepartseg_all_pts.py --model="pointnet2_part_seg_msg" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2025-07-30_15-09_rgbi_new_sampling" --rgbi_channel

python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_treepartseg_all_pts_INTERPOLATE_PROB_CORRECT_VERSION.py --model="pointnet2_part_seg_msg" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2025-11-18_19-21_rgbi_new_sampling_latest_dataset" --rgbi_channel





python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_treepartseg_all_pts_INTERPOLATE_PROB_CORRECT_VERSION_with_background.py --model="pointnet2_part_seg_msg" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2025-11-19_16-46_rgbi_new_sampling_latest_dataset_with_background" --rgbi_channel


# trained on 2048 pts, test on 2048 pts followed by interpolation
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_treepartseg_2048_pts_INTERPOLATE_PROB_CORRECT_VERSION_with_background_with_ignore.py --model="pointnet2_part_seg_msg" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2025-12-01_17-55_rgbi_new_sampling_latest_dataset_with_background_with_ignore" --rgbi_channel


# trained on all pts, test on all pts
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_treepartseg_ALL_PTS_with_background_with_ignore.py --model="pointnet2_part_seg_msg" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2025-12-02_15-53_rgbi_new_sampling_latest_dataset_with_background_with_ignore_ALL_PTS" --rgbi_channel

# trained using new batching idea, test on all pts
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_treepartseg_ALL_PTS_with_background_with_ignore.py --model="pointnet2_part_seg_msg" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2025-12-06_15-00_rgbi_with_background_new_batching" --rgbi_channel

# trained using fps on all pts 8192 with background, test on all pts
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_treepartseg_ALL_PTS_with_background_with_ignore.py --model="pointnet2_part_seg_msg" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2025-12-08_13-01_rgbi_with_background_fps_8192_pts" --rgbi_channel


# trained using class fps on all pts 6000 with background, test on all pts
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_treepartseg_ALL_PTS_with_background_with_ignore.py --model="pointnet2_part_seg_msg" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2025-12-08_16-57_rgbi_with_background_class_fps_6000_pts" --rgbi_channel


# trained using fps on all pts 8192 with background DICE LOSS, test on all pts
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_treepartseg_ALL_PTS_with_background_with_ignore.py --model="pointnet2_part_seg_msg" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2025-12-09_16-29_rgbi_with_background_fps_8192_pts_DICE_LOSS" --rgbi_channel



# trained using FPS SPLIT 6000 PTS WITH OVERLAP WITH PADDING COMBO LOSS, test on 6000 pts splitted - not merged
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_6000_pts_overlap_with_padding.py --model="pointnet2_part_seg_msg" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2025-12-11_21-17_rgbi_600_pts_overlap_padding_COMBO_LOSS" --rgbi_channel

# trained using FPS SPLIT 6000 PTS WITH OVERLAP WITH PADDING COMBO LOSS, test on all pts
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_treepartseg_ALL_PTS_with_background_with_ignore.py --model="pointnet2_part_seg_msg" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2025-12-11_17-24_rgbi_all_pts_combo_loss" --rgbi_channel



# trained using FPS SPLIT 6000 PTS WITH OVERLAP WITH PADDING COMBO LOSS, test on 6000 pts followed by interpolation
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_treepartseg_2048_pts_INTERPOLATE_PROB_CORRECT_VERSION_with_background_with_ignore.py --model="pointnet2_part_seg_msg" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2025-12-11_21-17_rgbi_600_pts_overlap_padding_COMBO_LOSS" --num_point=6000 --rgbi_channel

# trained using FPS SPLIT 6000 PTS WITH OVERLAP WITH PADDING FOCAL LOSS, test on 6000 pts followed by interpolation
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_treepartseg_2048_pts_INTERPOLATE_PROB_CORRECT_VERSION_with_background_with_ignore.py --model="pointnet2_part_seg_msg" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2025-12-22_11-31_rgbi_600_pts_overlap_padding_FOCAL_LOSS" --num_point=6000 --rgbi_channel




# trained using FPS SPLIT 6000 PTS WITH OVERLAP WITH PADDING FOCAL LOSS + SHAPE_PRIOR_IGNORE_POINT_LAMBDA_0.4, test on 6000 pts followed by interpolation
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_treepartseg_2048_pts_INTERPOLATE_PROB_CORRECT_VERSION_with_background_with_ignore.py --model="pointnet2_part_seg_msg" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2025-12-31_09-32_rgbi_600_pts_overlap_padding_FOCAL_LOSS_SHAPE_PRIOR_IGNORE_POINT_LAMBDA_0.4" --num_point=6000 --rgbi_channel



# trained using FPS SPLIT 6000 PTS WITH OVERLAP WITH PADDING FOCAL LOSS + SHAPE_PRIOR_LAMBDA_0.4, test on 6000 pts followed by interpolation
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_treepartseg_2048_pts_INTERPOLATE_PROB_CORRECT_VERSION_with_background_with_ignore.py --model="pointnet2_part_seg_msg" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2025-12-30_13-32_rgbi_600_pts_overlap_padding_FOCAL_LOSS_SHAPE_PRIOR_LAMBDA_0.4" --num_point=6000 --rgbi_channel









# trained using fps on all pts 8192 with background DICE LOSS, test on 8192 FPS points (splitted - not merged)
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_treepartseg_8192_FPS_PTS_EACH_with_background_with_ignore.py --model="pointnet2_part_seg_msg" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2025-12-09_16-29_rgbi_with_background_fps_8192_pts_DICE_LOSS" --rgbi_channel


# trained using fps on all pts 8192 with background COMBO LOSS, test on 8192 FPS points (splitted - not merged)
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_treepartseg_8192_FPS_PTS_EACH_with_background_with_ignore.py --model="pointnet2_part_seg_msg" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2025-12-11_13-35_rgbi_with_background_fps_8192_pts_COMBO_LOSS" --rgbi_channel


# trained on 2048 pts, test on all pts
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_treepartseg_all_pts_INTERPOLATE_PROB_CORRECT_VERSION_with_background_with_ignore.py --model="pointnet2_part_seg_msg" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2025-12-01_17-55_rgbi_new_sampling_latest_dataset_with_background_with_ignore" --rgbi_channel


# trained on all pts, test on all pts (only tree points, no background)
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_treepartseg_all_points_batch_1.py --model="pointnet2_part_seg_msg" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2025-12-03_14-49_rgbi_all_points_tree_only_batch_1" --rgbi_channel
    



# trained on all pts FOCAL LOSS, test on all pts
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_treepartseg_ALL_PTS_with_background_with_ignore.py --model="pointnet2_part_seg_msg" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-01-10_15-19_rgbi_all_pts_focal_loss_300_eps" --rgbi_channel




# trained using FPS SPLIT 6000 PTS WITH OVERLAP WITH PADDING FOCAL LOSS, test on 6000 pts followed by interpolation, FG - BG only
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_6000_points_with_interpolation_fg_bg.py --model="pointnet2_part_seg_msg" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-01-13_21-39_rgbi_6000_pts_overlap_padding_FOCAL_LOSS_FG_BG_ALPHA_2_.5" --num_point=6000 --rgbi_channel


# trained using FPS SPLIT 6000 PTS WITH OVERLAP WITH PADDING FOCAL LOSS, test on 6000 pts followed by interpolation, FG - BG only
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_6000_points_with_interpolation_fg_bg.py --model="pointnet2_part_seg_msg" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-01-13_21-26_rgbi_6000_pts_overlap_padding_FOCAL_LOSS_FG_BG_ALPHA_2_05" --num_point=6000 --rgbi_channel

python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_6000_points_with_interpolation_fg_bg_group_pred.py --model="pointnet2_part_seg_msg" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-01-13_21-26_rgbi_6000_pts_overlap_padding_FOCAL_LOSS_FG_BG_ALPHA_2_05" --num_point=6000 --rgbi_channel

python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_6000_points_with_interpolation_fg_bg_group_pred.py --model="pointnet2_part_seg_msg" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-01-13_21-39_rgbi_6000_pts_overlap_padding_FOCAL_LOSS_FG_BG_ALPHA_2_.5" --num_point=6000 --rgbi_channel


# trained using FPS SPLIT 12k PTS WITH OVERLAP WITH PADDING FOCAL LOSS, test on 12k pts followed by interpolation, FG - BG only
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_6000_points_with_interpolation_fg_bg.py --model="pointnet2_part_seg_msg" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-01-14_09-55_rgbi_12k_pts_overlap_padding_FOCAL_LOSS_FG_BG_ALPHA_2_.5" --num_point=12000 --rgbi_channel


# trained using FPS SPLIT 6000 PTS WITH OVERLAP WITH PADDING FOCAL LOSS, test on 6000 pts followed by interpolation, FG - BG only
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_6000_points_with_interpolation_fg_bg.py --model="pointnet2_part_seg_msg" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-01-14_10-42_rgbi_12k_pts_overlap_padding_FOCAL_LOSS_FG_BG_ALPHA_2_05" --num_point=12000 --rgbi_channel


# trained using FPS SPLIT 12k PTS WITH OVERLAP WITH PADDING FOCAL LOSS (Gaussian Wt), test on 12k pts followed by interpolation, FG - BG only
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_6000_points_with_interpolation_fg_bg.py --model="pointnet2_part_seg_msg" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-01-14_16-45_rgbi_12k_pts_overlap_padding_FOCAL_LOSS_FG_BG_ALPHA_None_Gaussian_Wt_True" --num_point=12000 --rgbi_channel


# trained using FPS SPLIT 6k PTS WITH OVERLAP WITH PADDING FOCAL LOSS (Gaussian Wt), test on 6000 pts followed by interpolation, FG - BG only
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_6000_points_with_interpolation_fg_bg_group_pred.py --model="pointnet2_part_seg_msg" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-01-14_19-54_rgbi_6k_pts_overlap_padding_FOCAL_LOSS_FG_BG_ALPHA_None_Gaussian_Wt_True" --num_point=6000 --rgbi_channel



# test for 600 trees
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_6000_points_with_interpolation_fg_bg_group_pred_600_trees.py --model="pointnet2_part_seg_msg" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-01-26_21-48_rgbi_4000_pts_overlap_padding_FOCAL_LOSS_FG_BG_ALPHA_None_Gaussian_Wt_True_600_trees" --num_point=4000 --rgbi_channel


# after removing some bad annotations
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_6000_points_with_interpolation_fg_bg_group_pred_600_trees.py --model="pointnet2_part_seg_msg" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-01-28_13-20_rgbi_4000_pts_overlap_padding_FOCAL_LOSS_FG_BG_ALPHA_None_Gaussian_Wt_True_600_trees_FOLD_0" --num_point=4000 --rgbi_channel











# trained on 60k pts FOCAL LOSS, test on 60k pts followed by interpolation or truncation
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_60k_pts_FG_BG.py --model="pointnet2_part_seg_msg" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-01-11_21-44_rgbi_60k_pts_focal_loss_fg_bg_distn_wt" --rgbi_channel

# trained on 60k pts FOCAL LOSS, test on 60k pts followed by interpolation or truncation
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_60k_pts_FG_BG.py --model="pointnet2_part_seg_msg" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-01-11_21-44_rgbi_60k_pts_focal_loss_fg_bg_distn_wt" --rgbi_channel

# trained on 60k pts FOCAL LOSS, test on 60k pts followed by interpolation or truncation
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_60k_pts_FG_BG.py --model="pointnet2_part_seg_msg" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-01-11_21-44_rgbi_60k_pts_focal_loss_fg_bg_distn_wt" --rgbi_channel






# Stage 2
# FPS SPLIT 4000 PTS WITH OVERLAP WITH PADDING FOCAL LOSS ONLY (alpha=None, Gaussian Wt=True),  : FG - BG only 
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_6000_points_with_interpolation_fg_bg_group_pred_stage_2.py --model="pointnet2_part_seg_msg" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-01-21_08-31_STAGE_2_rgbi_4000_pts_overlap_padding_FOCAL_LOSS_FG_BG_ALPHA_None_Gaussian_Wt_FALSE_thr30" --num_point=4000 --rgbi_channel


# FPS SPLIT 4000 PTS WITH OVERLAP WITH PADDING FOCAL LOSS ONLY (alpha=None, Gaussian Wt=True),  : FG - BG only THRESHOLD 30
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_6000_points_with_interpolation_fg_bg_group_pred_stage_2_thr30.py --model="pointnet2_part_seg_msg" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-01-21_08-31_STAGE_2_rgbi_4000_pts_overlap_padding_FOCAL_LOSS_FG_BG_ALPHA_None_Gaussian_Wt_FALSE_thr30" --num_point=4000 --rgbi_channel






















python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_treepartseg_all_pts.py --model="pointnet2_part_seg_msg" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2025-07-30_14-55_xyz_new_sampling"


# XYZ only - 120 standing, 60 fallen
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_treepartseg_all_pts_v2.py --model="pointnet2_part_seg_msg" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2025-07-30_18-02_rgbi_new_sampling_new_interpolation" --rgbi_channel


python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_treepartseg_all_pts_v2.py --model="pointnet2_part_seg_msg" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2025-07-30_18-00_xyz_new_sampling_new_interpolation"


python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_treepartseg.py --model="pointnet2_part_seg_msg" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2025-08-04_18-15_xyz_all_points_orig_code"


python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_treepartseg.py --model="pointnet2_part_seg_msg" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2025-08-04_16-13_rgbi_all_points_orig_code" --rgbi_channel