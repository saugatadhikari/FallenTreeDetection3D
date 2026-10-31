# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_treepartseg_repeat_pts.py --epoch=50

# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_treepartseg_repeat_pts_v2.py --epoch=300 --model="pointnet2_part_seg_msg" --rgbi_channel

# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_treepartseg_all_points.py --epoch=300 --model="pointnet2_part_seg_msg" --tag="rgbi_new_sampling" --rgbi_channel

python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_treepartseg_all_points.py --epoch=100 --model="pointnet2_part_seg_msg" --tag="xyz_new_sampling_latest_dataset"

python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_treepartseg_all_points.py --epoch=100 --model="pointnet2_part_seg_msg" --tag="rgbi_new_sampling_latest_dataset" --rgbi_channel



python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_treepartseg_all_points_batch_1.py --epoch=50 --model="pointnet2_part_seg_msg" --tag="rgbi_all_points_tree_only_batch_1" --rgbi_channel




python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_treepartseg_all_points_with_background.py --epoch=50 --model="pointnet2_part_seg_msg" --tag="rgbi_new_sampling_latest_dataset_with_background" --rgbi_channel






python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_treepartseg_2048_points_with_background_with_ignore.py.py --epoch=50 --model="pointnet2_part_seg_msg" --tag="rgbi_new_sampling_latest_dataset_with_background_with_ignore_2048_PTS" --rgbi_channel --batch_size 16

python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_treepartseg_all_points_with_background_with_ignore.py --epoch=50 --model="pointnet2_part_seg_msg" --tag="rgbi_new_sampling_latest_dataset_with_background_with_ignore_ALL_PTS" --rgbi_channel --batch_size 16



# USING NEW BATCHING IDEA
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_treepartseg_BATCH_SAMPLE_PTS.py --epoch=50 --model="pointnet2_part_seg_msg" --tag="rgbi_with_background_new_batching" --rgbi_channel


# FPS SPLIT 8192 PTS
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_treepartseg_8192_points_with_background_with_ignore.py --epoch=100 --model="pointnet2_part_seg_msg" --tag="rgbi_with_background_fps_8192_pts_COMBO_LOSS" --rgbi_channel --batch_size=8 --npoint=8192


# FPS SPLIT 6000 PTS WITH OVERLAP WITH PADDING COMBO LOSS
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_overlap_with_padding.py --epoch=200 --model="pointnet2_part_seg_msg" --tag="rgbi_600_pts_overlap_padding_COMBO_LOSS" --rgbi_channel --batch_size=8 --npoint=6000


# FPS SPLIT 6000 PTS WITH OVERLAP WITH PADDING FOCAL LOSS ONLY
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_overlap_with_padding.py --epoch=200 --model="pointnet2_part_seg_msg" --tag="rgbi_600_pts_overlap_padding_FOCAL_LOSS" --rgbi_channel --batch_size=8 --npoint=6000


# FPS CLASS SPLIT 6000 PTS - root ball, pit all points
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_treepartseg_8192_points_with_background_with_ignore.py --epoch=50 --model="pointnet2_part_seg_msg" --tag="rgbi_with_background_class_fps_6000_pts" --rgbi_channel --batch_size=8 --npoint=6000



# FPS SPLIT 6000 PTS WITH OVERLAP WITH PADDING FOCAL LOSS + SHAPE_PRIOR Loss (no ignore point)
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_overlap_with_padding_shape_prior.py --epoch=200 --model="pointnet2_part_seg_msg" --tag="rgbi_600_pts_overlap_padding_FOCAL_LOSS_SHAPE_PRIOR_LAMBDA_0.4" --rgbi_channel --batch_size=8 --npoint=6000 --lambda_shape="0.4"


# FPS SPLIT 6000 PTS WITH OVERLAP WITH PADDING FOCAL LOSS + SHAPE_PRIOR Loss (with ignore point)
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_overlap_with_padding_shape_prior.py --epoch=200 --model="pointnet2_part_seg_msg" --tag="rgbi_600_pts_overlap_padding_FOCAL_LOSS_SHAPE_PRIOR_IGNORE_POINT_LAMBDA_0.4" --rgbi_channel --batch_size=8 --npoint=6000 --lambda_shape="0.4"


# ALL PTS - FOCAL LOSS 
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_treepartseg_all_points_with_background_with_ignore.py --epoch=300 --model="pointnet2_part_seg_msg" --tag="rgbi_all_pts_focal_loss_300_eps" --rgbi_channel --batch_size=1 --npoint=6000


# 60k PTS - Focal Loss with fixed weights (2, 1.0) - FG vs BG only - 
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_60k_pts_FG_BG.py --epoch=300 --model="pointnet2_part_seg_msg" --tag="rgbi_60k_pts_focal_loss_fg_bg_distn_wt" --rgbi_channel --batch_size=8 --npoint=60000



# 60k PTS - Focal Loss with weights based on distance (boundary more weights) + alpha=[2.0, 1.0] - FG vs BG only - 
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_60k_pts_FG_BG.py --epoch=300 --model="pointnet2_part_seg_msg" --tag="rgbi_60k_pts_focal_loss_fg_bg_gaussian_wt" --rgbi_channel --batch_size=8 --npoint=60000 --learning_rate="0.001"

# 60k PTS - Focal Loss with weights based on distance (boundary more weights) (alpha=None) - FG vs BG only - 
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_60k_pts_FG_BG.py --epoch=300 --model="pointnet2_part_seg_msg" --tag="rgbi_60k_pts_focal_loss_fg_bg_gaussian_wt_alpha_none" --rgbi_channel --batch_size=8 --npoint=60000 --learning_rate="0.001"


# 60k PTS - Focal Loss (Gaussian wt=True, alpha=None) - ALL PARTS
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_60k_pts.py --epoch=200 --model="pointnet2_part_seg_msg" --tag="rgbi_60k_pts_focal_loss_all_parts_gaussian_wt_alpha_none" --rgbi_channel --batch_size=8 --npoint=60000 --learning_rate="0.001" --use_intensity=0

# 60k PTS - Focal Loss (Gaussian wt=None, alpha=[2,2,2,0.05]) - ALL PARTS
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_60k_pts.py --epoch=200 --model="pointnet2_part_seg_msg" --tag="rgbi_60k_pts_focal_loss_all_parts_alpha_2221" --rgbi_channel --batch_size=8 --npoint=60000 --learning_rate="0.001" --use_intensity=0

# 60k PTS - Focal Loss (Gaussian wt=None, alpha=[2,2,2,0.05]) - ALL PARTS
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_60k_pts.py --epoch=200 --model="pointnet2_part_seg_msg_60k" --tag="rgbi_60k_pts_focal_loss_all_parts_alpha_22205" --rgbi_channel --batch_size=8 --npoint=60000 --learning_rate="0.001" --use_intensity=0




# FPS SPLIT 6000 PTS WITH OVERLAP WITH PADDING FOCAL LOSS ONLY (alpha=None),  : FG - BG only 
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg.py --epoch=200 --model="pointnet2_part_seg_msg" --tag="rgbi_6000_pts_overlap_padding_FOCAL_LOSS_FG_BG" --rgbi_channel --batch_size=8 --npoint=6000

# FPS SPLIT 6000 PTS WITH OVERLAP WITH PADDING FOCAL LOSS ONLY (alpha=[2,0.05]),  : FG - BG only 
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg.py --epoch=200 --model="pointnet2_part_seg_msg" --tag="rgbi_6000_pts_overlap_padding_FOCAL_LOSS_FG_BG_ALPHA_2_05" --rgbi_channel --batch_size=8 --npoint=6000

# FPS SPLIT 6000 PTS WITH OVERLAP WITH PADDING FOCAL LOSS ONLY (alpha=[2,0.5]),  : FG - BG only 
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg.py --epoch=200 --model="pointnet2_part_seg_msg" --tag="rgbi_6000_pts_overlap_padding_FOCAL_LOSS_FG_BG_ALPHA_2_.5" --rgbi_channel --batch_size=8 --npoint=6000


# FPS SPLIT 12k PTS WITH OVERLAP WITH PADDING FOCAL LOSS ONLY (alpha=[2,0.5]),  : FG - BG only 
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg.py --epoch=200 --model="pointnet2_part_seg_msg" --tag="rgbi_12k_pts_overlap_padding_FOCAL_LOSS_FG_BG_ALPHA_2_.5" --rgbi_channel --batch_size=8 --npoint=12000

# FPS SPLIT 12k PTS WITH OVERLAP WITH PADDING FOCAL LOSS ONLY (alpha=[2,0.05]),  : FG - BG only 
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg.py --epoch=200 --model="pointnet2_part_seg_msg" --tag="rgbi_12k_pts_overlap_padding_FOCAL_LOSS_FG_BG_ALPHA_2_05" --rgbi_channel --batch_size=8 --npoint=12000

# FPS SPLIT 12k PTS WITH OVERLAP WITH PADDING FOCAL LOSS ONLY (alpha=None, Gaussian Wt=True),  : FG - BG only 
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg.py --epoch=200 --model="pointnet2_part_seg_msg" --tag="rgbi_12k_pts_overlap_padding_FOCAL_LOSS_FG_BG_ALPHA_None_Gaussian_Wt_True" --rgbi_channel --batch_size=8 --npoint=12000



# FPS SPLIT 6k PTS WITH OVERLAP WITH PADDING FOCAL LOSS ONLY (alpha=None, Gaussian Wt=True),  : FG - BG only 
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg.py --epoch=200 --model="pointnet2_part_seg_msg" --tag="rgbi_6k_pts_overlap_padding_FOCAL_LOSS_FG_BG_ALPHA_None_Gaussian_Wt_True" --rgbi_channel --batch_size=8 --npoint=6000


# FPS SPLIT 2048 PTS WITH OVERLAP WITH PADDING FOCAL LOSS ONLY (alpha=None, Gaussian Wt=True),  : FG - BG only 
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg.py --epoch=200 --model="pointnet2_part_seg_msg" --tag="rgbi_2048_pts_overlap_padding_FOCAL_LOSS_FG_BG_ALPHA_None_Gaussian_Wt_True" --rgbi_channel --batch_size=8 --npoint=2048

# FPS SPLIT 3000 PTS WITH OVERLAP WITH PADDING FOCAL LOSS ONLY (alpha=None, Gaussian Wt=True),  : FG - BG only 
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg.py --epoch=200 --model="pointnet2_part_seg_msg" --tag="rgbi_3000_pts_overlap_padding_FOCAL_LOSS_FG_BG_ALPHA_None_Gaussian_Wt_True" --rgbi_channel --batch_size=8 --npoint=3000

# FPS SPLIT 4000 PTS WITH OVERLAP WITH PADDING FOCAL LOSS ONLY (alpha=None, Gaussian Wt=True),  : FG - BG only 
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg.py --epoch=200 --model="pointnet2_part_seg_msg" --tag="rgbi_4000_pts_overlap_padding_FOCAL_LOSS_FG_BG_ALPHA_None_Gaussian_Wt_True" --rgbi_channel --batch_size=8 --npoint=4000

# FPS SPLIT 5000 PTS WITH OVERLAP WITH PADDING FOCAL LOSS ONLY (alpha=None, Gaussian Wt=True),  : FG - BG only 
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg.py --epoch=200 --model="pointnet2_part_seg_msg" --tag="rgbi_5000_pts_overlap_padding_FOCAL_LOSS_FG_BG_ALPHA_None_Gaussian_Wt_True" --rgbi_channel --batch_size=8 --npoint=5000


# FPS SPLIT 4000 PTS WITH OVERLAP WITH PADDING FOCAL LOSS ONLY (alpha=None, Gaussian Wt=True),  : FG - BG only 
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg.py --epoch=200 --model="pointnet2_part_seg_msg" --tag="rgbi_4000_pts_overlap_padding_FOCAL_LOSS_FG_BG_ALPHA_None_Gaussian_Wt_True" --rgbi_channel --batch_size=8 --npoint=4000




# FPS SPLIT 4000 PTS WITH OVERLAP WITH PADDING FOCAL LOSS ONLY (alpha=None, Gaussian Wt=True),  : FG - BG only; 600 trees
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_600_trees_ignore_handled.py --epoch=200 --model="pointnet2_part_seg_msg" --tag="rgbi_4000_pts_overlap_padding_FOCAL_LOSS_FG_BG_ALPHA_None_Gaussian_Wt_True_600_trees" --rgbi_channel --batch_size=8 --npoint=4000

python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_600_trees_no_ignore.py --epoch=200 --model="pointnet2_part_seg_msg" --tag="rgbi_4000_pts_overlap_padding_FOCAL_LOSS_FG_BG_ALPHA_None_Gaussian_Wt_True_600_trees_NO_IGNORE" --rgbi_channel --batch_size=8 --npoint=4000




# multi-fold train-test
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_600_trees_ignore_handled.py --epoch=200 --model="pointnet2_part_seg_msg" --tag="rgbi_4000_pts_overlap_padding_FOCAL_LOSS_FG_BG_ALPHA_None_Gaussian_Wt_True_600_trees_FOLD_1" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=1

python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_600_trees_ignore_handled.py --epoch=200 --model="pointnet2_part_seg_msg" --tag="rgbi_4000_pts_overlap_padding_FOCAL_LOSS_FG_BG_ALPHA_None_Gaussian_Wt_True_600_trees_FOLD_2" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=2

python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_600_trees_ignore_handled.py --epoch=200 --model="pointnet2_part_seg_msg" --tag="rgbi_4000_pts_overlap_padding_FOCAL_LOSS_FG_BG_ALPHA_None_Gaussian_Wt_True_600_trees_FOLD_3" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=3

python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_600_trees_ignore_handled.py --epoch=200 --model="pointnet2_part_seg_msg" --tag="rgbi_4000_pts_overlap_padding_FOCAL_LOSS_FG_BG_ALPHA_None_Gaussian_Wt_True_600_trees_FOLD_4" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=4





# Stage 2
# FPS SPLIT 4000 PTS WITH OVERLAP WITH PADDING FOCAL LOSS ONLY (alpha=None, Gaussian Wt=True),  : FG - BG only 
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_stage_2.py --epoch=200 --model="pointnet2_part_seg_msg" --tag="STAGE_2_rgbi_4000_pts_overlap_padding_FOCAL_LOSS_FG_BG_ALPHA_None_Gaussian_Wt_True_YES" --rgbi_channel --batch_size=8 --npoint=4000


python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_stage_2.py --epoch=200 --model="pointnet2_part_seg_msg" --tag="STAGE_2_rgbi_4000_pts_overlap_padding_FOCAL_LOSS_FG_BG_ALPHA_None_Gaussian_Wt_True_YES_thr30" --rgbi_channel --batch_size=8 --npoint=4000


python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_stage_2.py --epoch=200 --model="pointnet2_part_seg_msg" --tag="STAGE_2_rgbi_4000_pts_overlap_padding_FOCAL_LOSS_FG_BG_ALPHA_None_Gaussian_Wt_FALSE_thr30" --rgbi_channel --batch_size=8 --npoint=4000



























# FPS SPLIT 6k PTS WITH OVERLAP WITH PADDING FOCAL LOSS ONLY (log cosh dice),  : FG - BG only 
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_logcoshdice.py --epoch=200 --model="pointnet2_part_seg_msg" --tag="rgbi_6k_pts_overlap_padding_LOG_COSH_DICE_FG_BG" --rgbi_channel --batch_size=8 --npoint=6000





python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_treepartseg_all_points_with_background.py --epoch=50 --model="pointnet2_part_seg_msg" --tag="rgbi_new_sampling_latest_dataset_with_background_with_ignore_ALL_PTS" --rgbi_channel




# python train_treepartseg_repeat_pts.py --epoch=100

python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_treepartseg_all_points_MINI_POINTNET_v2.py --epoch=100 --model="pointnet2_part_seg_msg" --tag="rgbi_new_sampling_new_interpolation" --rgbi_channel

python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_treepartseg_all_points_MINI_POINTNET_v2.py --epoch=100 --model="pointnet2_part_seg_msg" --tag="xyz_new_sampling_new_interpolation"








python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_treepartseg.py --epoch=100 --model="pointnet2_part_seg_msg" --tag="xyz_all_points_orig_code" --batch_size=1


python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_treepartseg.py --epoch=100 --model="pointnet2_part_seg_msg" --tag="rgbi_all_points_orig_code" --batch_size=1 --rgbi_channel