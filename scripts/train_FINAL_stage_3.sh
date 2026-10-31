# conda activate pointnext

# if alpha_fg=0, we don't use gaussian weight, all points get wt=1

# Fold 0; train larger model; Sigma 1.0, alpha=None, intensity=0, augment=3
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_stage_3_FINAL.py --epoch=100 --model="pointnet2_part_seg_msg_4k_stage_3" --tag="STAGE_3_FOLD_0_FINAL_POLYGONS" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=0 --sigma="0.8" --num_aug_copies=3 --alpha_fg="2"

# Fold 1; train larger model; Sigma 1.0, alpha=None, intensity=0, augment=3
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_stage_3_FINAL.py --epoch=100 --model="pointnet2_part_seg_msg_4k_stage_3" --tag="STAGE_3_FOLD_1_FINAL_POLYGONS" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=1 --sigma="0.8" --num_aug_copies=3 --alpha_fg="2"

# Fold 2; train larger model; Sigma 1.0, alpha=None, intensity=0, augment=3
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_stage_3_FINAL.py --epoch=100 --model="pointnet2_part_seg_msg_4k_stage_3" --tag="STAGE_3_FOLD_2_FINAL_POLYGONS" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=2 --sigma="0.8" --num_aug_copies=3 --alpha_fg="2"

# Fold 3; train larger model; Sigma 1.0, alpha=None, intensity=0, augment=3
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_stage_3_FINAL.py --epoch=100 --model="pointnet2_part_seg_msg_4k_stage_3" --tag="STAGE_3_FOLD_3_FINAL_POLYGONS" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=3 --sigma="0.8" --num_aug_copies=3 --alpha_fg="2"

# Fold 4; train larger model; Sigma 1.0, alpha=None, intensity=0, augment=3
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_stage_3_FINAL.py --epoch=100 --model="pointnet2_part_seg_msg_4k_stage_3" --tag="STAGE_3_FOLD_4_FINAL_POLYGONS" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=4 --sigma="0.8" --num_aug_copies=3 --alpha_fg="2"



# same wt
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_stage_3_FINAL.py --epoch=100 --model="pointnet2_part_seg_msg_4k_stage_3" --tag="STAGE_3_FOLD_0_FINAL_POLYGONS_SAME_WT" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=0 --sigma="0.8" --num_aug_copies=3 --alpha_fg=2 --same_weight=1

python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_stage_3_FINAL.py --epoch=100 --model="pointnet2_part_seg_msg_4k_stage_3" --tag="STAGE_3_FOLD_1_FINAL_POLYGONS_SAME_WT" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=1 --sigma="0.8" --num_aug_copies=3 --alpha_fg=2 --same_weight=1

python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_stage_3_FINAL.py --epoch=100 --model="pointnet2_part_seg_msg_4k_stage_3" --tag="STAGE_3_FOLD_2_FINAL_POLYGONS_SAME_WT" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=2 --sigma="0.8" --num_aug_copies=3 --alpha_fg=2 --same_weight=1

python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_stage_3_FINAL.py --epoch=100 --model="pointnet2_part_seg_msg_4k_stage_3" --tag="STAGE_3_FOLD_3_FINAL_POLYGONS_SAME_WT" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=3 --sigma="0.8" --num_aug_copies=3 --alpha_fg=2 --same_weight=1

python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_stage_3_FINAL.py --epoch=100 --model="pointnet2_part_seg_msg_4k_stage_3" --tag="STAGE_3_FOLD_4_FINAL_POLYGONS_SAME_WT" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=4 --sigma="0.8" --num_aug_copies=3 --alpha_fg=2 --same_weight=1



