# Fold 1; train larger model; Sigma 0.8, alpha=None, intensity=0, augment=3
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_600_trees_ignore_handled_no_intensity.py --epoch=100 --model="pointnet2_part_seg_msg_4k" --tag="rgb_4k_5_blocks_FG_BG_ALPHA_None_Gaussian_Wt_True_FOLD_0_SIGMA_0.8" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=1 --sigma="0.8" --num_aug_copies=3

# Fold 1; train larger model; Sigma 1.0, alpha=None, intensity=0, augment=3
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_600_trees_ignore_handled_no_intensity.py --epoch=100 --model="pointnet2_part_seg_msg_4k" --tag="rgb_4k_5_blocks_FG_BG_ALPHA_None_Gaussian_Wt_True_FOLD_0_SIGMA_1.0" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=1 --sigma="1.0" --num_aug_copies=3

# Fold 1; train larger model; Sigma 1.2, alpha=None, intensity=0, augment=3
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_600_trees_ignore_handled_no_intensity.py --epoch=100 --model="pointnet2_part_seg_msg_4k" --tag="rgb_4k_5_blocks_FG_BG_ALPHA_None_Gaussian_Wt_True_FOLD_0_SIGMA_1.2" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=1 --sigma="1.2" --num_aug_copies=3