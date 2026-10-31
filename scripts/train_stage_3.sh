# fold 0, same wt
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_stage_3.py --epoch=100 --model="pointnet2_part_seg_msg_4k_stage_3" --tag="STAGE_3_FOLD_0_SAME_WT" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=0 --sigma="0.8" --num_aug_copies=3 --alpha_fg=2 --same_weight=1

# fold 1, same wt
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_stage_3.py --epoch=100 --model="pointnet2_part_seg_msg_4k_stage_3" --tag="STAGE_3_FOLD_1_SAME_WT" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=1 --sigma="0.8" --num_aug_copies=3 --alpha_fg=2 --same_weight=1

# fold 2, same wt
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_stage_3.py --epoch=100 --model="pointnet2_part_seg_msg_4k_stage_3" --tag="STAGE_3_FOLD_2_SAME_WT" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=2 --sigma="0.8" --num_aug_copies=3 --alpha_fg=2 --same_weight=1

# fold 3, same wt
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_stage_3.py --epoch=100 --model="pointnet2_part_seg_msg_4k_stage_3" --tag="STAGE_3_FOLD_3_SAME_WT" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=3 --sigma="0.8" --num_aug_copies=3 --alpha_fg=2 --same_weight=1

# fold 4, same wt
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_stage_3.py --epoch=100 --model="pointnet2_part_seg_msg_4k_stage_3" --tag="STAGE_3_FOLD_4_SAME_WT" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=4 --sigma="0.8" --num_aug_copies=3 --alpha_fg=2 --same_weight=1




# fold 0, alpha=20_1
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_stage_3.py --epoch=100 --model="pointnet2_part_seg_msg_4k_stage_3" --tag="STAGE_3_FOLD_0_ALPHA_20_1" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=0 --sigma="0.8" --num_aug_copies=3 --alpha_fg=2 --same_weight=0

# fold 1, alpha=20_1
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_stage_3.py --epoch=100 --model="pointnet2_part_seg_msg_4k_stage_3" --tag="STAGE_3_FOLD_1_ALPHA_20_1" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=1 --sigma="0.8" --num_aug_copies=3 --alpha_fg=2 --same_weight=0





# fold 0, alpha=None
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_stage_3.py --epoch=100 --model="pointnet2_part_seg_msg_4k_stage_3" --tag="STAGE_3_FOLD_0_ALPHA_None" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=0 --sigma="0.8" --num_aug_copies=3 --same_weight=0

# fold 1, alpha=20_1
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_stage_3.py --epoch=100 --model="pointnet2_part_seg_msg_4k_stage_3" --tag="STAGE_3_FOLD_1_ALPHA_None" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=1 --sigma="0.8" --num_aug_copies=3 --same_weight=0



# fold 1, single stage alpha=20_1
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_single_stage.py --epoch=100 --model="pointnet2_part_seg_msg_4k_stage_3" --tag="SINGLE_STAGE_FOLD_1_ALPHA_8_1" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=1 --sigma="0.8" --num_aug_copies=3 --alpha_fg="0.8" --same_weight=1






# Stage 3: after SSU (synthetic data added) ... !!! ...

# fold 0, same wt
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_stage_3.py --epoch=100 --model="pointnet2_part_seg_msg_4k_stage_3" --tag="FOLD_0_STAGE_3_SYN_DATA" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=0 --sigma="0.8" --num_aug_copies=3 --alpha_fg=2 --same_weight=1

# fold 1, same wt
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_stage_3.py --epoch=100 --model="pointnet2_part_seg_msg_4k_stage_3" --tag="FOLD_1_STAGE_3_SYN_DATA" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=1 --sigma="0.8" --num_aug_copies=3 --alpha_fg=2 --same_weight=1

# fold 2, same wt
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_stage_3.py --epoch=100 --model="pointnet2_part_seg_msg_4k_stage_3" --tag="FOLD_2_STAGE_3_SYN_DATA" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=2 --sigma="0.8" --num_aug_copies=3 --alpha_fg=2 --same_weight=1

# fold 3, same wt
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_stage_3.py --epoch=100 --model="pointnet2_part_seg_msg_4k_stage_3" --tag="FOLD_3_STAGE_3_SYN_DATA" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=3 --sigma="0.8" --num_aug_copies=3 --alpha_fg=2 --same_weight=1

# fold 4, same wt
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_stage_3.py --epoch=100 --model="pointnet2_part_seg_msg_4k_stage_3" --tag="FOLD_4_STAGE_3_SYN_DATA" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=4 --sigma="0.8" --num_aug_copies=3 --alpha_fg=2 --same_weight=1