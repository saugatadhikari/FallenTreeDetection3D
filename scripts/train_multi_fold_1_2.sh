# # python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_600_trees_ignore_handled.py --epoch=200 --model="pointnet2_part_seg_msg" --tag="rgbi_4000_pts_overlap_padding_FOCAL_LOSS_FG_BG_ALPHA_None_Gaussian_Wt_True_600_trees_FOLD_0" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=0

# # python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_600_trees_ignore_handled.py --epoch=200 --model="pointnet2_part_seg_msg" --tag="rgbi_4000_pts_overlap_padding_FOCAL_LOSS_FG_BG_ALPHA_None_Gaussian_Wt_True_600_trees_FOLD_0_NO_IGNORE_ON_VAL" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=0

# # don't use intensity
# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_600_trees_ignore_handled_no_intensity.py --epoch=200 --model="pointnet2_part_seg_msg" --tag="rgb_4000_pts_overlap_padding_FOCAL_LOSS_FG_BG_ALPHA_None_Gaussian_Wt_True_600_trees_FOLD_0_NO_INTENSITY" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=0

# ALL BLOCKS
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_600_trees_ignore_handled.py --epoch=100 --model="pointnet2_part_seg_msg" --tag="rgbi_4000_pts_ALL_blocks_FOCAL_LOSS_FG_BG_ALPHA_None_Gaussian_Wt_True_600_trees_FOLD_0_SIGMA_1.0" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=0 --sigma="1.0"

# ALL BLOCKS
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_600_trees_ignore_handled.py --epoch=100 --model="pointnet2_part_seg_msg" --tag="rgbi_4000_pts_ALL_blocks_FOCAL_LOSS_FG_BG_ALPHA_None_Gaussian_Wt_True_600_trees_FOLD_0_SIGMA_0.8_crossover_1" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=0 --sigma="0.8"

python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_600_trees_ignore_handled.py --epoch=100 --model="pointnet2_part_seg_msg" --tag="rgbi_4000_pts_ALL_blocks_FOCAL_LOSS_FG_BG_ALPHA_None_Gaussian_Wt_True_600_trees_FOLD_0_SIGMA_1.0_crossover_1_factor_1" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=0 --sigma="1.0"

python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_600_trees_ignore_handled.py --epoch=100 --model="pointnet2_part_seg_msg" --tag="rgbi_4000_pts_ALL_blocks_FOCAL_LOSS_FG_BG_ALPHA_None_Gaussian_Wt_True_600_trees_FOLD_0_SIGMA_1.0_FG_only_wt" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=0 --sigma="1.0"

python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_600_trees_ignore_handled.py --epoch=100 --model="pointnet2_part_seg_msg" --tag="rgbi_4000_pts_ALL_blocks_FOCAL_LOSS_FG_BG_Gaussian_Wt_True_600_trees_FOLD_0_SIGMA_1.0_ALPHA_5_1" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=0 --sigma="1.0"

# sigma=0.8, all blocks, alpha=10_1
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_600_trees_ignore_handled.py --epoch=100 --model="pointnet2_part_seg_msg" --tag="rgbi_4000_pts_ALL_blocks_FOCAL_LOSS_FG_BG_Gaussian_Wt_True_600_trees_FOLD_0_SIGMA_0.8_ALPHA_10_1" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=0 --sigma="1.0"



# SINGLE BLOCKS
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_600_trees_ignore_handled_single_block.py --epoch=200 --model="pointnet2_part_seg_msg" --tag="rgbi_4000_pts_single_block_FOCAL_LOSS_FG_BG_ALPHA_None_Gaussian_Wt_True_600_trees_FOLD_0_SIGMA_1.0" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=0 --sigma="1.0"

python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_600_trees_ignore_handled_single_block.py --epoch=200 --model="pointnet2_part_seg_msg" --tag="rgbi_4000_pts_single_block_FOCAL_LOSS_FG_BG_ALPHA_None_Gaussian_Wt_True_600_trees_FOLD_0_SIGMA_1.0_REPEAT_5" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=0 --sigma="1.0" --repeat_factor=5

python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_600_trees_ignore_handled_single_block.py --epoch=200 --model="pointnet2_part_seg_msg" --tag="rgbi_4000_pts_single_block_FOCAL_LOSS_FG_BG_ALPHA_None_Gaussian_Wt_True_600_trees_FOLD_0_SIGMA_1.0_REPEAT_1_SUB_1" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=0 --sigma="1.0" --repeat_factor=1 --subtraction=1

python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_600_trees_ignore_handled_single_block_with_aug.py --epoch=200 --model="pointnet2_part_seg_msg" --tag="rgbi_4000_pts_single_block_FOCAL_LOSS_FG_BG_ALPHA_None_Gaussian_Wt_True_600_trees_FOLD_0_SIGMA_1.0_REPEAT_0_SUB_0_AUG_3" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=0 --sigma="1.0" --repeat_factor=1 --subtraction=0

# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_600_trees_ignore_handled_single_block_val_miou.py --epoch=200 --model="pointnet2_part_seg_msg" --tag="rgbi_4000_pts_single_block_FOCAL_LOSS_FG_BG_ALPHA_None_Gaussian_Wt_True_600_trees_FOLD_0_VAL_MIOU" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=0


# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_600_trees_ignore_handled.py --epoch=200 --model="pointnet2_part_seg_msg" --tag="rgbi_4000_pts_overlap_padding_FOCAL_LOSS_FG_BG_ALPHA_None_Gaussian_Wt_True_600_trees_FOLD_1" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=1

# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_600_trees_ignore_handled.py --epoch=200 --model="pointnet2_part_seg_msg" --tag="rgbi_4000_pts_overlap_padding_FOCAL_LOSS_FG_BG_ALPHA_None_Gaussian_Wt_True_600_trees_FOLD_2" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=2


# # stage 2
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_stage_2.py --epoch=200 --model="pointnet2_part_seg_msg" --tag="rgbi_4000_pts_overlap_padding_FOCAL_LOSS_FG_BG_ALPHA_None_Gaussian_Wt_FALSE_600_trees_STAGE_2_FOLD_0_LATEST" --rgbi_channel --batch_size=8 --npoint=4000

python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_stage_2.py --epoch=200 --model="pointnet2_part_seg_msg" --tag="rgbi_4000_pts_overlap_padding_FOCAL_LOSS_FG_BG_ALPHA_Noneeee_Gaussian_Wt_FALSE_600_trees_STAGE_2_FOLD_0_LATEST_alpha_none" --rgbi_channel --batch_size=8 --npoint=4000

# post train from stage 1 for stage 2
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/post_train_6000_pts_fg_bg_stage_2.py --epoch=100 --model="pointnet2_part_seg_msg" --tag="rgbi_4000_pts_overlap_padding_FOCAL_LOSS_FG_BG_ALPHA_Noneeee_Gaussian_Wt_FALSE_600_trees_STAGE_2_FOLD_0_POST_TRAIN_FROM_STAGE_1" --rgbi_channel --batch_size=8 --npoint=4000

# post train from stage 1 for stage 2 - fixed lr - 0.001
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/post_train_6000_pts_fg_bg_stage_2_fixed_lr.py --epoch=100 --model="pointnet2_part_seg_msg" --tag="rgbi_4000_pts_overlap_padding_FOCAL_LOSS_FG_BG_ALPHA_Noneeee_Gaussian_Wt_FALSE_600_trees_STAGE_2_FOLD_0_POST_TRAIN_FROM_STAGE_1_FIXED_LR" --rgbi_channel --batch_size=8 --npoint=4000

# train for stage 2 using 4 layers starting with 2048 instead of 512
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_stage_2_4layer.py --epoch=100 --model="pointnet2_part_seg_msg_4k" --tag="rgbi_4000_pts_overlap_padding_FOCAL_LOSS_FG_BG_ALPHA_Noneeee_Gaussian_Wt_FALSE_600_trees_STAGE_2_FOLD_0_LATEST_alpha_none_4k" --rgbi_channel --batch_size=8 --npoint=4000



# # post-train on difficult cases
# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/post_train_6000_pts_fg_bg_600_trees_ignore_handled.py --epoch=300 --model="pointnet2_part_seg_msg" --log_dir="2026-01-26_21-48_rgbi_4000_pts_overlap_padding_FOCAL_LOSS_FG_BG_ALPHA_None_Gaussian_Wt_True_600_trees" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=0

python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_600_trees_ignore_handled.py --epoch=100 --model="pointnet2_part_seg_msg_4k" --tag="rgbi_4000_pts_overlap_padding_FOCAL_LOSS_FG_BG_ALPHA_None_Gaussian_Wt_True_600_trees_FOLD_0_4k" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=0


python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_600_trees_ignore_handled.py --epoch=150 --model="pointnet2_part_seg_msg" --tag="rgbi_4000_pts_overlap_padding_FOCAL_LOSS_FG_BG_ALPHA_None_Gaussian_Wt_True_600_trees_FOLD_1_LATEST" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=1

python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_6000_pts_fg_bg_600_trees_ignore_handled.py --epoch=150 --model="pointnet2_part_seg_msg" --tag="rgbi_4000_pts_overlap_padding_FOCAL_LOSS_FG_BG_ALPHA_None_Gaussian_Wt_True_600_trees_FOLD_2_LATEST" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=2