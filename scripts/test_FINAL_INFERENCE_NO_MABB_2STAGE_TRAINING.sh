# Using models trained on 2 stages only (No MABB - simple polygons)

# # Stage 1 From 2D - NoMABB - R111
# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_stage_1_FINAL_NoMABB.py --model="pointnet2_part_seg_msg_4k" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="BEST_MODELS_FOR_2D_STAGE_1_NO_MABB" --input_folder="/home/adhiksa@iu.edu/Research/PointCloud/Data/TreePartAnnotation/data/trees_from_2d_R111_Stage1_LATEST/" --output_folder="R111_OUTPUT_FROM_2D_STAGE_1_LATEST_ONLY_POLYGONS_NO_MABB" --num_point=4000 --rgbi_channel --fold_id=0

# # Stage 1 From 2D - NoMABB - R112
# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_stage_1_FINAL_NoMABB.py --model="pointnet2_part_seg_msg_4k" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="BEST_MODELS_FOR_2D_STAGE_1_NO_MABB" --input_folder="/home/adhiksa@iu.edu/Research/PointCloud/Data/TreePartAnnotation/data/trees_from_2d_R112_Stage1_LATEST/" --output_folder="R112_OUTPUT_FROM_2D_STAGE_1_LATEST_ONLY_POLYGONS_NO_MABB" --num_point=4000 --rgbi_channel --fold_id=0

# # Stage 1 From 2D - NoMABB - R132
# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_stage_1_FINAL_NoMABB.py --model="pointnet2_part_seg_msg_4k" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="BEST_MODELS_FOR_2D_STAGE_1_NO_MABB" --input_folder="/home/adhiksa@iu.edu/Research/PointCloud/Data/TreePartAnnotation/data/trees_from_2d_R132_Stage1_LATEST/" --output_folder="R132_OUTPUT_FROM_2D_STAGE_1_LATEST_ONLY_POLYGONS_NO_MABB" --num_point=4000 --rgbi_channel --fold_id=0



# Stage 3 From 2D - NoMABB - R111 - No Stage 2 - Thr30 in stage 1
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_stage_3_FINAL_NoMABB_NoStage2.py --model="pointnet2_part_seg_msg_4k_stage_3" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="BEST_MODELS_FOR_2D_STAGE_3_NO_MABB" --input_folder="/home/adhiksa@iu.edu/Research/PointCloud/Data/TreePartAnnotation/data/trees_from_2d_R111_Stage2_LATEST_ONLY_POLYGONS_NO_MABB/" --output_folder="R111_OUTPUT_FROM_2D_STAGE_3_LATEST_NO_Stage2_Thr30S1_OrigTrainNoMABB" --num_point=4000 --rgbi_channel --fold_id=0

# Stage 3 From 2D - NoMABB - R112 - No Stage 2 - Thr30 in stage 1
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_stage_3_FINAL_NoMABB_NoStage2.py --model="pointnet2_part_seg_msg_4k_stage_3" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="BEST_MODELS_FOR_2D_STAGE_3_NO_MABB" --input_folder="/home/adhiksa@iu.edu/Research/PointCloud/Data/TreePartAnnotation/data/trees_from_2d_R112_Stage2_LATEST_ONLY_POLYGONS_NO_MABB/" --output_folder="R112_OUTPUT_FROM_2D_STAGE_3_LATEST_NO_Stage2_Thr30S1_OrigTrainNoMABB" --num_point=4000 --rgbi_channel --fold_id=0

# # Stage 3 From 2D - NoMABB - R132 - No Stage 2 - Thr30 in stage 1
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_stage_3_FINAL_NoMABB_NoStage2.py --model="pointnet2_part_seg_msg_4k_stage_3" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="BEST_MODELS_FOR_2D_STAGE_3_NO_MABB" --input_folder="/home/adhiksa@iu.edu/Research/PointCloud/Data/TreePartAnnotation/data/trees_from_2d_R132_Stage2_LATEST_ONLY_POLYGONS_NO_MABB/" --output_folder="R132_OUTPUT_FROM_2D_STAGE_3_LATEST_NO_Stage2_Thr30S1_OrigTrainNoMABB" --num_point=4000 --rgbi_channel --fold_id=0

