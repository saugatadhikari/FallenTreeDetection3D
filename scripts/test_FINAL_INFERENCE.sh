

# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_stage_1_FINAL.py --model="pointnet2_part_seg_msg_4k" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-04-27_13-03_FOLD_0_FINAL" --num_point=4000 --rgbi_channel --fold_id=0


# # previously trained model: Fold 0
# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_stage_1_FINAL.py --model="pointnet2_part_seg_msg_4k" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-03-04_22-12_rgb_4k_5_blocks_FG_BG_ALPHA_20_1_Gaussian_Wt_True_FOLD_0_SIGMA_0.8" --num_point=4000 --rgbi_channel --fold_id=0


# # previously trained model: Fold 1
# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_stage_1_FINAL.py --model="pointnet2_part_seg_msg_4k" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-02-27_09-09_rgb_4k_5_blocks_FG_BG_ALPHA_20_1_Gaussian_Wt_True_FOLD_1_SIGMA_0.8" --num_point=4000 --rgbi_channel --fold_id=0


# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_stage_1_FINAL.py --model="pointnet2_part_seg_msg_4k" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-04-27_22-53_FOLD_0_FINAL_ALPHA_40_1_AUG_0" --num_point=4000 --rgbi_channel --fold_id=0



# Stage 1 From 2D
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_stage_1_FINAL.py --model="pointnet2_part_seg_msg_4k" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="BEST_MODELS_FOR_2D_STAGE_1" --num_point=4000 --rgbi_channel --fold_id=0

# Stage 2 From 2D
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_stage_2_FINAL.py --model="pointnet2_part_seg_msg_4k" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="BEST_MODELS_FOR_2D_STAGE_2" --num_point=4000 --rgbi_channel --fold_id=0


# Stage 1_2 From 2D - NoMABB
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_stage_1_2_FINAL.py --model="pointnet2_part_seg_msg_4k" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="BEST_MODELS_FOR_2D_STAGE_1_2_NoMABB" --num_point=4000 --rgbi_channel --fold_id=0







# Stage 1 From 2D - NoMABB - R111
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_stage_1_FINAL_NoMABB.py --model="pointnet2_part_seg_msg_4k" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="BEST_MODELS_FOR_2D_STAGE_1" --num_point=4000 --rgbi_channel --fold_id=0

# Stage 1 From 2D - NoMABB - R112
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_stage_1_FINAL_NoMABB.py --model="pointnet2_part_seg_msg_4k" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --input_folder="/home/adhiksa@iu.edu/Research/PointCloud/Data/TreePartAnnotation/data/trees_from_2d_R112_stage_1_NoMABB/" --output_folder="R112_OUTPUT_FROM_2D_STAGE_1" --log_dir="BEST_MODELS_FOR_2D_STAGE_1" --num_point=4000 --rgbi_channel --fold_id=0

# Stage 1 From 2D - NoMABB - R132
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_stage_1_FINAL_NoMABB.py --model="pointnet2_part_seg_msg_4k" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --input_folder="/home/adhiksa@iu.edu/Research/PointCloud/Data/TreePartAnnotation/data/trees_from_2d_R132_stage_1_NoMABB/" --output_folder="R132_OUTPUT_FROM_2D_STAGE_1" --log_dir="BEST_MODELS_FOR_2D_STAGE_1" --num_point=4000 --rgbi_channel --fold_id=0




# Stage 2 From 2D - NoMABB
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_stage_2_FINAL_NoMABB.py --model="pointnet2_part_seg_msg_4k" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="BEST_MODELS_FOR_2D_STAGE_2" --num_point=4000 --rgbi_channel --fold_id=0

# Stage 3 From 2D - NoMABB
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_stage_3_FINAL_NoMABB.py --model="pointnet2_part_seg_msg_4k_stage_3" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="BEST_MODELS_FOR_2D_STAGE_3" --num_point=4000 --rgbi_channel --fold_id=0






# Stage 2 From 2D - NoMABB - No Stage 1
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_stage_2_FINAL_NoMABB_NoStage1.py --model="pointnet2_part_seg_msg_4k" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="BEST_MODELS_FOR_2D_STAGE_2" --output_folder="OUTPUT_FROM_2D_STAGE_2_NO_STAGE_1" --num_point=4000 --rgbi_channel --fold_id=0

# Stage 3 From 2D - NoMABB - No Stage 1
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_stage_3_FINAL_NoMABB_NoStage1.py --model="pointnet2_part_seg_msg_4k_stage_3" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="BEST_MODELS_FOR_2D_STAGE_3" --output_folder="OUTPUT_FROM_2D_STAGE_3_NO_STAGE_1" --num_point=4000 --rgbi_channel --fold_id=0




# Stage 3 From 2D - NoMABB - No Stage 2
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_stage_3_FINAL_NoMABB_NoStage2.py --model="pointnet2_part_seg_msg_4k_stage_3" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="BEST_MODELS_FOR_2D_STAGE_3" --output_folder="OUTPUT_FROM_2D_STAGE_3_NO_STAGE_2" --num_point=4000 --rgbi_channel --fold_id=0

# Stage 3 From 2D - NoMABB - No Stage 2 - R112
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_stage_3_FINAL_NoMABB_NoStage2.py --model="pointnet2_part_seg_msg_4k_stage_3" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="BEST_MODELS_FOR_2D_STAGE_3" --input_folder="/home/adhiksa@iu.edu/Research/PointCloud/Data/TreePartAnnotation/data/trees_from_2d_R112_stage_2_NoMABB/" --output_folder="R112_OUTPUT_FROM_2D_STAGE_3_NO_STAGE_2" --num_point=4000 --rgbi_channel --fold_id=0

# Stage 3 From 2D - NoMABB - No Stage 2 - R132
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_stage_3_FINAL_NoMABB_NoStage2.py --model="pointnet2_part_seg_msg_4k_stage_3" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="BEST_MODELS_FOR_2D_STAGE_3" --input_folder="/home/adhiksa@iu.edu/Research/PointCloud/Data/TreePartAnnotation/data/trees_from_2d_R132_stage_2_NoMABB/" --output_folder="R132_OUTPUT_FROM_2D_STAGE_3_NO_STAGE_2" --num_point=4000 --rgbi_channel --fold_id=0



# LATEST after 05/18 - thr 30 in stage 1

# Stage 2 From 2D - NoMABB - R111
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_stage_2_FINAL_NoMABB.py --model="pointnet2_part_seg_msg_4k" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="BEST_MODELS_FOR_2D_STAGE_2" --input_folder="/home/adhiksa@iu.edu/Research/PointCloud/Data/TreePartAnnotation/data/trees_from_2d_R111_stage_2_NoMABB/" --output_folder="R111_OUTPUT_FROM_2D_STAGE_2_LATEST_THR30(Stage1)" --num_point=4000 --rgbi_channel --fold_id=0

# Stage 2 From 2D - NoMABB - R112
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_stage_2_FINAL_NoMABB.py --model="pointnet2_part_seg_msg_4k" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="BEST_MODELS_FOR_2D_STAGE_2" --input_folder="/home/adhiksa@iu.edu/Research/PointCloud/Data/TreePartAnnotation/data/trees_from_2d_R112_stage_2_NoMABB/" --output_folder="R112_OUTPUT_FROM_2D_STAGE_2_LATEST_THR30(Stage1)" --num_point=4000 --rgbi_channel --fold_id=0

# Stage 2 From 2D - NoMABB - R132
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_stage_2_FINAL_NoMABB.py --model="pointnet2_part_seg_msg_4k" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="BEST_MODELS_FOR_2D_STAGE_2" --input_folder="/home/adhiksa@iu.edu/Research/PointCloud/Data/TreePartAnnotation/data/trees_from_2d_R132_stage_2_NoMABB/" --output_folder="R132_OUTPUT_FROM_2D_STAGE_2_LATEST_THR30(Stage1)" --num_point=4000 --rgbi_channel --fold_id=0


# NEW MERGE with new thresholds and PCA issue fix: 05/18

# Stage 1 From 2D - NoMABB - R111
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_stage_1_FINAL_NoMABB.py --model="pointnet2_part_seg_msg_4k" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="BEST_MODELS_FOR_2D_STAGE_1" --input_folder="/home/adhiksa@iu.edu/Research/PointCloud/Data/TreePartAnnotation/data/trees_from_2d_R111_Stage1_LATEST/" --output_folder="R111_OUTPUT_FROM_2D_STAGE_1_LATEST" --num_point=4000 --rgbi_channel --fold_id=0

# Stage 1 From 2D - NoMABB - R112
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_stage_1_FINAL_NoMABB.py --model="pointnet2_part_seg_msg_4k" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="BEST_MODELS_FOR_2D_STAGE_1" --input_folder="/home/adhiksa@iu.edu/Research/PointCloud/Data/TreePartAnnotation/data/trees_from_2d_R112_Stage1_LATEST/" --output_folder="R112_OUTPUT_FROM_2D_STAGE_1_LATEST" --num_point=4000 --rgbi_channel --fold_id=0

# Stage 1 From 2D - NoMABB - R132
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_stage_1_FINAL_NoMABB.py --model="pointnet2_part_seg_msg_4k" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="BEST_MODELS_FOR_2D_STAGE_1" --input_folder="/home/adhiksa@iu.edu/Research/PointCloud/Data/TreePartAnnotation/data/trees_from_2d_R132_Stage1_LATEST/" --output_folder="R132_OUTPUT_FROM_2D_STAGE_1_LATEST" --num_point=4000 --rgbi_channel --fold_id=0


# Stage 3 From 2D - NoMABB - R111 - No Stage 2
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_stage_3_FINAL_NoMABB_NoStage2.py --model="pointnet2_part_seg_msg_4k_stage_3" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="BEST_MODELS_FOR_2D_STAGE_3" --input_folder="/home/adhiksa@iu.edu/Research/PointCloud/Data/TreePartAnnotation/data/trees_from_2d_R111_Stage2_LATEST/" --output_folder="R111_OUTPUT_FROM_2D_STAGE_3_LATEST_NO_Stage2" --num_point=4000 --rgbi_channel --fold_id=0

# Stage 3 From 2D - NoMABB - R112 - No Stage 2
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_stage_3_FINAL_NoMABB_NoStage2.py --model="pointnet2_part_seg_msg_4k_stage_3" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="BEST_MODELS_FOR_2D_STAGE_3" --input_folder="/home/adhiksa@iu.edu/Research/PointCloud/Data/TreePartAnnotation/data/trees_from_2d_R112_Stage2_LATEST/" --output_folder="R112_OUTPUT_FROM_2D_STAGE_3_LATEST_NO_Stage2" --num_point=4000 --rgbi_channel --fold_id=0

# Stage 3 From 2D - NoMABB - R132 - No Stage 2
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_stage_3_FINAL_NoMABB_NoStage2.py --model="pointnet2_part_seg_msg_4k_stage_3" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="BEST_MODELS_FOR_2D_STAGE_3" --input_folder="/home/adhiksa@iu.edu/Research/PointCloud/Data/TreePartAnnotation/data/trees_from_2d_R132_Stage2_LATEST/" --output_folder="R132_OUTPUT_FROM_2D_STAGE_3_LATEST_NO_Stage2" --num_point=4000 --rgbi_channel --fold_id=0


# NEW MERGE with new thresholds and PCA issue fix: 05/18 - thr30 in stage1

# Stage 2 From 2D - NoMABB - R111
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_stage_2_FINAL_NoMABB.py --model="pointnet2_part_seg_msg_4k" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="BEST_MODELS_FOR_2D_STAGE_2" --input_folder="/home/adhiksa@iu.edu/Research/PointCloud/Data/TreePartAnnotation/data/trees_from_2d_R111_Stage2_LATEST_thr30_in_stage1/" --output_folder="R111_OUTPUT_FROM_2D_STAGE_2_LATEST_THR30_IN_STAGE1" --num_point=4000 --rgbi_channel --fold_id=0

# Stage 2 From 2D - NoMABB - R112
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_stage_2_FINAL_NoMABB.py --model="pointnet2_part_seg_msg_4k" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="BEST_MODELS_FOR_2D_STAGE_2" --input_folder="/home/adhiksa@iu.edu/Research/PointCloud/Data/TreePartAnnotation/data/trees_from_2d_R112_Stage2_LATEST_thr30_in_stage1/" --output_folder="R112_OUTPUT_FROM_2D_STAGE_2_LATEST_THR30_IN_STAGE1" --num_point=4000 --rgbi_channel --fold_id=0

# Stage 2 From 2D - NoMABB - R132
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_stage_2_FINAL_NoMABB.py --model="pointnet2_part_seg_msg_4k" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="BEST_MODELS_FOR_2D_STAGE_2" --input_folder="/home/adhiksa@iu.edu/Research/PointCloud/Data/TreePartAnnotation/data/trees_from_2d_R132_Stage2_LATEST_thr30_in_stage1/" --output_folder="R132_OUTPUT_FROM_2D_STAGE_2_LATEST_THR30_IN_STAGE1" --num_point=4000 --rgbi_channel --fold_id=0

# NEW MERGE with new thresholds and PCA issue fix: 05/18 - thr30 in stage1 thr30 in stage2

# Stage 3 From 2D - NoMABB - R111
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_stage_3_FINAL_NoMABB.py --model="pointnet2_part_seg_msg_4k_stage_3" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="BEST_MODELS_FOR_2D_STAGE_3" --input_folder="/home/adhiksa@iu.edu/Research/PointCloud/Data/TreePartAnnotation/data/trees_from_2d_R111_Stage3_LATEST_Thr30S1_Thr30S2/" --output_folder="R111_OUTPUT_FROM_2D_STAGE_3_LATEST_THR30S1_THR30S2" --num_point=4000 --rgbi_channel --fold_id=0

# Stage 3 From 2D - NoMABB - R112
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_stage_3_FINAL_NoMABB.py --model="pointnet2_part_seg_msg_4k_stage_3" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="BEST_MODELS_FOR_2D_STAGE_3" --input_folder="/home/adhiksa@iu.edu/Research/PointCloud/Data/TreePartAnnotation/data/trees_from_2d_R112_Stage3_LATEST_Thr30S1_Thr30S2/" --output_folder="R112_OUTPUT_FROM_2D_STAGE_3_LATEST_THR30S1_THR30S2" --num_point=4000 --rgbi_channel --fold_id=0

# Stage 3 From 2D - NoMABB - R132
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_stage_3_FINAL_NoMABB.py --model="pointnet2_part_seg_msg_4k_stage_3" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="BEST_MODELS_FOR_2D_STAGE_3" --input_folder="/home/adhiksa@iu.edu/Research/PointCloud/Data/TreePartAnnotation/data/trees_from_2d_R132_Stage3_LATEST_Thr30S1_Thr30S2/" --output_folder="R132_OUTPUT_FROM_2D_STAGE_3_LATEST_THR30S1_THR30S2" --num_point=4000 --rgbi_channel --fold_id=0



# Stage 3 From 2D - NoMABB - R111 - No Stage 2 - Thr30 in stage 1
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_stage_3_FINAL_NoMABB_NoStage2.py --model="pointnet2_part_seg_msg_4k_stage_3" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="BEST_MODELS_FOR_2D_STAGE_3" --input_folder="/home/adhiksa@iu.edu/Research/PointCloud/Data/TreePartAnnotation/data/trees_from_2d_R111_Stage2_LATEST_thr30_in_stage1/" --output_folder="R111_OUTPUT_FROM_2D_STAGE_3_LATEST_NO_Stage2_Thr30S1" --num_point=4000 --rgbi_channel --fold_id=0

# Stage 3 From 2D - NoMABB - R112 - No Stage 2 - Thr30 in stage 1
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_stage_3_FINAL_NoMABB_NoStage2.py --model="pointnet2_part_seg_msg_4k_stage_3" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="BEST_MODELS_FOR_2D_STAGE_3" --input_folder="/home/adhiksa@iu.edu/Research/PointCloud/Data/TreePartAnnotation/data/trees_from_2d_R112_Stage2_LATEST_thr30_in_stage1/" --output_folder="R112_OUTPUT_FROM_2D_STAGE_3_LATEST_NO_Stage2_Thr30S1" --num_point=4000 --rgbi_channel --fold_id=0

# Stage 3 From 2D - NoMABB - R132 - No Stage 2 - Thr30 in stage 1
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_stage_3_FINAL_NoMABB_NoStage2.py --model="pointnet2_part_seg_msg_4k_stage_3" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="BEST_MODELS_FOR_2D_STAGE_3" --input_folder="/home/adhiksa@iu.edu/Research/PointCloud/Data/TreePartAnnotation/data/trees_from_2d_R132_Stage2_LATEST_thr30_in_stage1/" --output_folder="R132_OUTPUT_FROM_2D_STAGE_3_LATEST_NO_Stage2_Thr30S1" --num_point=4000 --rgbi_channel --fold_id=0




# # Using models trained on 2 stages only (No MABB - simple polygons)

# # Stage 1 From 2D - NoMABB - R111
# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_stage_1_FINAL_NoMABB.py --model="pointnet2_part_seg_msg_4k" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="BEST_MODELS_FOR_2D_STAGE_1_NO_MABB" --input_folder="/home/adhiksa@iu.edu/Research/PointCloud/Data/TreePartAnnotation/data/trees_from_2d_R111_Stage1_LATEST/" --output_folder="R111_OUTPUT_FROM_2D_STAGE_1_LATEST_ONLY_POLYGONS_NO_MABB" --num_point=4000 --rgbi_channel --fold_id=0

# # Stage 1 From 2D - NoMABB - R112
# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_stage_1_FINAL_NoMABB.py --model="pointnet2_part_seg_msg_4k" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="BEST_MODELS_FOR_2D_STAGE_1_NO_MABB" --input_folder="/home/adhiksa@iu.edu/Research/PointCloud/Data/TreePartAnnotation/data/trees_from_2d_R112_Stage1_LATEST/" --output_folder="R112_OUTPUT_FROM_2D_STAGE_1_LATEST_ONLY_POLYGONS_NO_MABB" --num_point=4000 --rgbi_channel --fold_id=0

# # Stage 1 From 2D - NoMABB - R132
# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_stage_1_FINAL_NoMABB.py --model="pointnet2_part_seg_msg_4k" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="BEST_MODELS_FOR_2D_STAGE_1_NO_MABB" --input_folder="/home/adhiksa@iu.edu/Research/PointCloud/Data/TreePartAnnotation/data/trees_from_2d_R132_Stage1_LATEST/" --output_folder="R132_OUTPUT_FROM_2D_STAGE_1_LATEST_ONLY_POLYGONS_NO_MABB" --num_point=4000 --rgbi_channel --fold_id=0

