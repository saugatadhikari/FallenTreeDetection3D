## --------- PART SEG CODE RUNNING LIKE SEM SEG ---------- ##

# trained on 60k pts FOCAL LOSS (gaussian wt = False, alpha = [2,1]), test on 60k pts followed by interpolation or truncation
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_60k_pts_FG_BG.py --model="pointnet2_part_seg_msg" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-01-11_21-44_rgbi_60k_pts_focal_loss_fg_bg_distn_wt" --rgbi_channel

# trained on 60k pts FOCAL LOSS (gaussian wt = True, alpha=[2, 1]), test on 60k pts followed by interpolation or truncation
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_60k_pts_FG_BG.py --model="pointnet2_part_seg_msg" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-01-12_10-10_rgbi_60k_pts_focal_loss_fg_bg_gaussian_wt" --rgbi_channel

# trained on 60k pts FOCAL LOSS (gaussian wt = True, alpha=None), test on 60k pts followed by interpolation or truncation
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_60k_pts_FG_BG.py --model="pointnet2_part_seg_msg" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-01-12_10-21_rgbi_60k_pts_focal_loss_fg_bg_gaussian_wt_alpha_none" --rgbi_channel


## --------- PART SEG CODE RUNNING LIKE SEM SEG ---------- ##

# trained on 60k pts FOCAL LOSS (gaussian wt = False, alpha = [2,1]), test on ALL PTS
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_all_pts_FG_BG.py --model="pointnet2_part_seg_msg" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-01-11_21-44_rgbi_60k_pts_focal_loss_fg_bg_distn_wt" --rgbi_channel

# trained on 60k pts FOCAL LOSS (gaussian wt = True, alpha=[2, 1]), test on ALL PTS
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_all_pts_FG_BG.py --model="pointnet2_part_seg_msg" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-01-12_10-10_rgbi_60k_pts_focal_loss_fg_bg_gaussian_wt" --rgbi_channel

# trained on 60k pts FOCAL LOSS (gaussian wt = True, alpha=None), test on ALL PTS
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_all_pts_FG_BG.py --model="pointnet2_part_seg_msg" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-01-12_10-21_rgbi_60k_pts_focal_loss_fg_bg_gaussian_wt_alpha_none" --rgbi_channel


## --------- SEM SEG CODE ---------- ##

# trained on 60k pts FOCAL LOSS, test on 60k pts followed by interpolation or truncation
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_60k_pts_FG_BG.py --model="pointnet2_part_seg_msg" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-01-11_21-44_rgbi_60k_pts_focal_loss_fg_bg_distn_wt" --rgbi_channel

## --------- SEM SEG CODE ---------- ##

# trained on 60k pts FOCAL LOSS (gaussian wt = True, alpha=None), test on ALL PTS -- SEM SEG 
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_60k_pts_FG_BG_SEM_SEG.py --model="pointnet2_sem_seg_msg_60k" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-01-12_13-44_SEM_SEG_rgbi_60k_pts_focal_loss_fg_bg_gaussian_wt_alpha_none" --rgbi_channel

# trained on 60k pts FOCAL LOSS (gaussian wt = None, alpha=[2,1]), test on ALL PTS -- SEM SEG 
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_60k_pts_FG_BG_SEM_SEG.py --model="pointnet2_sem_seg_msg_60k" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-01-12_21-49_SEM_SEG_rgbi_60k_pts_focal_loss_fg_bg_alpha_2_1" --rgbi_channel