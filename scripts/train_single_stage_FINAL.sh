# Fold 0
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_single_stage_FINAL.py --epoch=100 --model="pointnet2_part_seg_msg_4k_stage_3" --tag="SINGLE_STAGE_FOLD_0_FINAL_POLYGONS_SAME_WT" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=0 --sigma="0.8" --num_aug_copies=3 --alpha_fg=2 --same_weight=1

# Fold 1
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_single_stage_FINAL.py --epoch=100 --model="pointnet2_part_seg_msg_4k_stage_3" --tag="SINGLE_STAGE_FOLD_1_FINAL_POLYGONS_SAME_WT" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=1 --sigma="0.8" --num_aug_copies=3 --alpha_fg=2 --same_weight=1

# Fold 2
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_single_stage_FINAL.py --epoch=100 --model="pointnet2_part_seg_msg_4k_stage_3" --tag="SINGLE_STAGE_FOLD_2_FINAL_POLYGONS_SAME_WT" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=2 --sigma="0.8" --num_aug_copies=3 --alpha_fg=2 --same_weight=1

# Fold 3
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_single_stage_FINAL.py --epoch=100 --model="pointnet2_part_seg_msg_4k_stage_3" --tag="SINGLE_STAGE_FOLD_3_FINAL_POLYGONS_SAME_WT" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=3 --sigma="0.8" --num_aug_copies=3 --alpha_fg=2 --same_weight=1

# Fold 4
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_single_stage_FINAL.py --epoch=100 --model="pointnet2_part_seg_msg_4k_stage_3" --tag="SINGLE_STAGE_FOLD_4_FINAL_POLYGONS_SAME_WT" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=4 --sigma="0.8" --num_aug_copies=3 --alpha_fg=2 --same_weight=1