

# 60k PTS - Focal Loss with weights based on distance (boundary more weights) (alpha=None) - FG vs BG only - 
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_tree_semseg_60k.py --epoch=200 --model="pointnet2_sem_seg_msg_60k" --tag="SEM_SEG_rgbi_60k_pts_focal_loss_fg_bg_gaussian_wt_alpha_none" --rgbi_channel --batch_size=8 --npoint=60000 --learning_rate="0.001"


# 60k PTS - Focal Loss with (alpha=[2,1]) - FG vs BG only - 
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_tree_semseg_60k.py --epoch=200 --model="pointnet2_sem_seg_msg_60k" --tag="SEM_SEG_rgbi_60k_pts_focal_loss_fg_bg_alpha_2_1" --rgbi_channel --batch_size=8 --npoint=60000 --learning_rate="0.001"

