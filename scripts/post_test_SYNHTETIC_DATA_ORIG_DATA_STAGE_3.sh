
# # Test

# # # Fold 0
# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/post_test_stage_1_SYNTHETIC_DATA_ORIG_DATA.py --model="pointnet2_part_seg_msg_4k" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-07-07_19-47_FOLD_0_SYN_PLUS_ORIG" --num_point=4000 --rgbi_channel --fold_id=0

# # # Fold 1
# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/post_test_stage_1_SYNTHETIC_DATA_ORIG_DATA.py --model="pointnet2_part_seg_msg_4k" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/scripts" --log_dir="2026-07-07_22-59_FOLD_1_SYN_PLUS_ORIG" --num_point=4000 --rgbi_channel --fold_id=1

# # Fold 2
# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/post_test_stage_1_SYNTHETIC_DATA_ORIG_DATA.py --model="pointnet2_part_seg_msg_4k" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/scripts" --log_dir="2026-07-08_10-08_FOLD_2_SYN_PLUS_ORIG" --num_point=4000 --rgbi_channel --fold_id=2

# # Fold 3
# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/post_test_stage_1_SYNTHETIC_DATA_ORIG_DATA.py --model="pointnet2_part_seg_msg_4k" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-07-06_13-57_FOLD_0_SYN_POST_TRAIN" --num_point=4000 --rgbi_channel --fold_id=3

# # # Fold 4
# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/post_test_stage_1_SYNTHETIC_DATA_ORIG_DATA.py --model="pointnet2_part_seg_msg_4k" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-07-08_10-42_FOLD_4_SYN_PLUS_ORIG" --num_point=4000 --rgbi_channel --fold_id=4



# ### kwaka kwaza

# # Fold 0
# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_stage_1_kwk.py --model="pointnet2_part_seg_msg_4k" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-02-27_09-09_rgb_4k_5_blocks_FG_BG_ALPHA_20_1_Gaussian_Wt_True_FOLD_1_SIGMA_0.8" --input_folder='/home/adhiksa@iu.edu/Research/PointCloud/Data/TreePartAnnotation/data/trees_600_min_bbox_with_ignore/' --output_folder="FOLD_0_STAGE_1_KWK_F1" --num_point=4000 --rgbi_channel --fold_id=0


# # Fold 1
# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_stage_1_kwk.py --model="pointnet2_part_seg_msg_4k" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-03-04_22-12_rgb_4k_5_blocks_FG_BG_ALPHA_20_1_Gaussian_Wt_True_FOLD_0_SIGMA_0.8" --input_folder='/home/adhiksa@iu.edu/Research/PointCloud/Data/TreePartAnnotation/data/trees_600_min_bbox_with_ignore/' --output_folder="FOLD_1_STAGE_1_KWK_F0" --num_point=4000 --rgbi_channel --fold_id=1


# # Fold 2
# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_stage_1_kwk.py --model="pointnet2_part_seg_msg_4k" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-03-04_09-57_rgb_4k_5_blocks_FG_BG_ALPHA_20_1_Gaussian_Wt_True_FOLD_4_SIGMA_0.8" --input_folder='/home/adhiksa@iu.edu/Research/PointCloud/Data/TreePartAnnotation/data/trees_600_min_bbox_with_ignore/' --output_folder="FOLD_2_STAGE_1_KWK_F4" --num_point=4000 --rgbi_channel --fold_id=2


# # Fold 3
# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_stage_1_kwk.py --model="pointnet2_part_seg_msg_4k" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-03-03_23-10_rgb_4k_5_blocks_FG_BG_ALPHA_20_1_Gaussian_Wt_True_FOLD_2_SIGMA_0.8_CROSSOVER_ALPHA" --input_folder='/home/adhiksa@iu.edu/Research/PointCloud/Data/TreePartAnnotation/data/trees_600_min_bbox_with_ignore/' --output_folder="FOLD_3_STAGE_1_KWK_F2" --num_point=4000 --rgbi_channel --fold_id=3


# # Fold 4
# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_stage_1_kwk.py --model="pointnet2_part_seg_msg_4k" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-03-04_09-57_rgb_4k_5_blocks_FG_BG_ALPHA_20_1_Gaussian_Wt_True_FOLD_3_SIGMA_0.8" --input_folder='/home/adhiksa@iu.edu/Research/PointCloud/Data/TreePartAnnotation/data/trees_600_min_bbox_with_ignore/' --output_folder="FOLD_3_STAGE_1_KWK_F3" --num_point=4000 --rgbi_channel --fold_id=4

# 


# Test Stage 2

# 2026-03-25_09-42_STAGE_2_FOLD_0_CW_FACTOR
# 2026-03-25_09-43_STAGE_2_FOLD_1_CW_FACTOR
# 2026-03-25_10-51_STAGE_2_FOLD_2_CW_FACTOR
# 2026-03-25_20-33_STAGE_2_FOLD_3_CW_FACTOR
# 2026-03-25_20-46_STAGE_2_FOLD_4_CW_FACTOR



# # Fold 0
# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_stage_2_kwk.py --model="pointnet2_part_seg_msg_4k" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-03-25_09-43_STAGE_2_FOLD_1_CW_FACTOR" --input_folder='/home/adhiksa@iu.edu/Research/PointCloud/Data/TreePartAnnotation/data/trees_600_from_stage_1/' --output_folder="FOLD_0_STAGE_2_KWK_F1" --num_point=4000 --rgbi_channel --fold_id=4

# # # Fold 1
# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_stage_2_kwk.py --model="pointnet2_part_seg_msg_4k" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-03-25_09-42_STAGE_2_FOLD_0_CW_FACTOR" --input_folder='/home/adhiksa@iu.edu/Research/PointCloud/Data/TreePartAnnotation/data/trees_600_from_stage_1/' --output_folder="FOLD_1_STAGE_2_KWK_F0" --num_point=4000 --rgbi_channel --fold_id=4

# # Fold 2
# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_stage_2_kwk.py --model="pointnet2_part_seg_msg_4k" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-03-25_20-46_STAGE_2_FOLD_4_CW_FACTOR" --input_folder='/home/adhiksa@iu.edu/Research/PointCloud/Data/TreePartAnnotation/data/trees_600_from_stage_1/' --output_folder="FOLD_2_STAGE_2_KWK_F4" --num_point=4000 --rgbi_channel --fold_id=4

# # Fold 3
# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_stage_2_kwk.py --model="pointnet2_part_seg_msg_4k" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-03-25_10-51_STAGE_2_FOLD_2_CW_FACTOR" --input_folder='/home/adhiksa@iu.edu/Research/PointCloud/Data/TreePartAnnotation/data/trees_600_from_stage_1/' --output_folder="FOLD_3_STAGE_2_KWK_F2" --num_point=4000 --rgbi_channel --fold_id=4

# # # Fold 4
# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_stage_2_kwk.py --model="pointnet2_part_seg_msg_4k" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-03-25_20-33_STAGE_2_FOLD_3_CW_FACTOR" --input_folder='/home/adhiksa@iu.edu/Research/PointCloud/Data/TreePartAnnotation/data/trees_600_from_stage_1/' --output_folder="FOLD_4_STAGE_2_KWK_F3" --num_point=4000 --rgbi_channel --fold_id=4


## kwaka kwaza Stage 3


# # Fold 0
# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_stage_3_kwk.py --model="pointnet2_part_seg_msg_4k_stage_3" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-03-27_17-36_STAGE_3_FOLD_1_SAME_WT" --input_folder='/home/adhiksa@iu.edu/Research/PointCloud/Data/TreePartAnnotation/data/trees_600_from_stage_2/' --output_folder="FOLD_0_STAGE_3_KWK_F1" --num_point=4000 --rgbi_channel --fold_id=0

# # # Fold 1
# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_stage_3_kwk.py --model="pointnet2_part_seg_msg_4k_stage_3" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-03-28_10-16_STAGE_3_FOLD_0_SAME_WT" --input_folder='/home/adhiksa@iu.edu/Research/PointCloud/Data/TreePartAnnotation/data/trees_600_from_stage_2/' --output_folder="FOLD_1_STAGE_3_KWK_F0" --num_point=4000 --rgbi_channel --fold_id=1

# # Fold 2
# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_stage_3_kwk.py --model="pointnet2_part_seg_msg_4k_stage_3" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-03-28_10-26_STAGE_3_FOLD_4_SAME_WT" --input_folder='/home/adhiksa@iu.edu/Research/PointCloud/Data/TreePartAnnotation/data/trees_600_from_stage_2/' --output_folder="FOLD_2_STAGE_3_KWK_F4" --num_point=4000 --rgbi_channel --fold_id=2

# # Fold 3
# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_stage_3_kwk.py --model="pointnet2_part_seg_msg_4k_stage_3" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-03-28_10-17_STAGE_3_FOLD_2_SAME_WT" --input_folder='/home/adhiksa@iu.edu/Research/PointCloud/Data/TreePartAnnotation/data/trees_600_from_stage_2/' --output_folder="FOLD_3_STAGE_3_KWK_F2" --num_point=4000 --rgbi_channel --fold_id=3

# # # Fold 4
# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_stage_3_kwk.py --model="pointnet2_part_seg_msg_4k_stage_3" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-03-28_10-20_STAGE_3_FOLD_3_SAME_WT" --input_folder='/home/adhiksa@iu.edu/Research/PointCloud/Data/TreePartAnnotation/data/trees_600_from_stage_2/' --output_folder="FOLD_4_STAGE_3_KWK_F3" --num_point=4000 --rgbi_channel --fold_id=4


# # Latest after SSU

# # # Fold 1
# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/post_test_stage_1_SYNTHETIC_DATA_ORIG_DATA.py --model="pointnet2_part_seg_msg_4k" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-08-28_18-41_FOLD_1_SYN_PLUS_ORIG" --num_point=4000 --rgbi_channel --fold_id=1


# # # # Fold 2
# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/post_test_stage_1_SYNTHETIC_DATA_ORIG_DATA.py --model="pointnet2_part_seg_msg_4k" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-08-28_18-41_FOLD_1_SYN_PLUS_ORIG" --num_point=4000 --rgbi_channel --fold_id=1


# # # # Fold 4
# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/post_test_stage_1_SYNTHETIC_DATA_ORIG_DATA.py --model="pointnet2_part_seg_msg_4k" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/scripts" --log_dir="2026-07-08_10-42_FOLD_4_SYN_PLUS_ORIG" --num_point=4000 --rgbi_channel --fold_id=4



# Stage 2 - after adding synthetic ... !!! ...
# # Fold 0
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_6000_points_with_interpolation_fg_bg_group_pred_stage_2.py --model="pointnet2_part_seg_msg_4k" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-09-25_14-10_FOLD_0_STAGE_2_SYN_DATA" --num_point=4000 --rgbi_channel --fold_id=0



# RUN THESE 2 !!!!!!!!!
# # Fold 1
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_6000_points_with_interpolation_fg_bg_group_pred_stage_2.py --model="pointnet2_part_seg_msg_4k" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-09-25_14-12_FOLD_1_STAGE_2_SYN_DATA" --num_point=4000 --rgbi_channel --fold_id=1

# Fold 2
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_6000_points_with_interpolation_fg_bg_group_pred_stage_2.py --model="pointnet2_part_seg_msg_4k" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-09-25_14-13_FOLD_2_STAGE_2_SYN_DATA" --num_point=4000 --rgbi_channel --fold_id=2




# Fold 3
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_6000_points_with_interpolation_fg_bg_group_pred_stage_2.py --model="pointnet2_part_seg_msg_4k" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-09-25_14-17_FOLD_3_STAGE_2_SYN_DATA" --num_point=4000 --rgbi_channel --fold_id=3

# # Fold 4
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_6000_points_with_interpolation_fg_bg_group_pred_stage_2.py --model="pointnet2_part_seg_msg_4k" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-09-25_17-50_FOLD_4_STAGE_2_SYN_DATA" --num_point=4000 --rgbi_channel --fold_id=4


# Stage 3 - after adding synthetic ... !!! ...
# # Fold 0
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_stage_3.py --model="pointnet2_part_seg_msg_4k_stage_3" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-09-27_00-11_FOLD_0_STAGE_3_SYN_DATA" --num_point=4000 --rgbi_channel --fold_id=0

# # Fold 1
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_stage_3.py --model="pointnet2_part_seg_msg_4k_stage_3" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-09-27_00-04_FOLD_1_STAGE_3_SYN_DATA" --num_point=4000 --rgbi_channel --fold_id=1

# Fold 2
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_stage_3.py --model="pointnet2_part_seg_msg_4k_stage_3" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-09-27_05-01_FOLD_2_STAGE_3_SYN_DATA" --num_point=4000 --rgbi_channel --fold_id=2

# Fold 3
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_stage_3.py --model="pointnet2_part_seg_msg_4k_stage_3" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-09-27_00-05_FOLD_3_STAGE_3_SYN_DATA" --num_point=4000 --rgbi_channel --fold_id=3

# # Fold 4
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_stage_3.py --model="pointnet2_part_seg_msg_4k_stage_3" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-09-27_00-06_FOLD_4_STAGE_3_SYN_DATA" --num_point=4000 --rgbi_channel --fold_id=4
