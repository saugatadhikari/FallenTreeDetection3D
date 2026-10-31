# NEW MERGE with new thresholds and PCA issue fix: 05/18

# # Stage 1 From 2D - NoMABB - R113
# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_stage_1_FINAL_NoMABB.py --model="pointnet2_part_seg_msg_4k" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="BEST_MODELS_FOR_2D_STAGE_1" --input_folder="/home/adhiksa@iu.edu/Research/PointCloud/Data/TreePartAnnotation/data/trees_from_2d_R113_Stage1_LATEST/" --output_folder="R113_OUTPUT_FROM_2D_STAGE_1_LATEST" --num_point=4000 --rgbi_channel --fold_id=0

# # Stage 1 From 2D - NoMABB - R121
# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_stage_1_FINAL_NoMABB.py --model="pointnet2_part_seg_msg_4k" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="BEST_MODELS_FOR_2D_STAGE_1" --input_folder="/home/adhiksa@iu.edu/Research/PointCloud/Data/TreePartAnnotation/data/trees_from_2d_R121_Stage1_LATEST/" --output_folder="R121_OUTPUT_FROM_2D_STAGE_1_LATEST" --num_point=4000 --rgbi_channel --fold_id=0

# # Stage 1 From 2D - NoMABB - R122
# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_stage_1_FINAL_NoMABB.py --model="pointnet2_part_seg_msg_4k" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="BEST_MODELS_FOR_2D_STAGE_1" --input_folder="/home/adhiksa@iu.edu/Research/PointCloud/Data/TreePartAnnotation/data/trees_from_2d_R122_Stage1_LATEST/" --output_folder="R122_OUTPUT_FROM_2D_STAGE_1_LATEST" --num_point=4000 --rgbi_channel --fold_id=0




# Stage 3 From 2D - NoMABB - R113 - No Stage 2 - Thr30 in stage 1
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_stage_3_FINAL_NoMABB_NoStage2.py --model="pointnet2_part_seg_msg_4k_stage_3" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="BEST_MODELS_FOR_2D_STAGE_3" --input_folder="/home/adhiksa@iu.edu/Research/PointCloud/Data/TreePartAnnotation/data/trees_from_2d_R113_Stage2_LATEST_thr30_in_stage1/" --output_folder="R113_OUTPUT_FROM_2D_STAGE_3_LATEST_NO_Stage2_Thr30S1" --num_point=4000 --rgbi_channel --fold_id=0

# Stage 3 From 2D - NoMABB - R121 - No Stage 2 - Thr30 in stage 1
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_stage_3_FINAL_NoMABB_NoStage2.py --model="pointnet2_part_seg_msg_4k_stage_3" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="BEST_MODELS_FOR_2D_STAGE_3" --input_folder="/home/adhiksa@iu.edu/Research/PointCloud/Data/TreePartAnnotation/data/trees_from_2d_R121_Stage2_LATEST_thr30_in_stage1/" --output_folder="R121_OUTPUT_FROM_2D_STAGE_3_LATEST_NO_Stage2_Thr30S1" --num_point=4000 --rgbi_channel --fold_id=0

# Stage 3 From 2D - NoMABB - R122 - No Stage 2 - Thr30 in stage 1
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_stage_3_FINAL_NoMABB_NoStage2.py --model="pointnet2_part_seg_msg_4k_stage_3" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="BEST_MODELS_FOR_2D_STAGE_3" --input_folder="/home/adhiksa@iu.edu/Research/PointCloud/Data/TreePartAnnotation/data/trees_from_2d_R122_Stage2_LATEST_thr30_in_stage1/" --output_folder="R122_OUTPUT_FROM_2D_STAGE_3_LATEST_NO_Stage2_Thr30S1" --num_point=4000 --rgbi_channel --fold_id=0

