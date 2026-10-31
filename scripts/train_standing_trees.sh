# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_standing_trees.py --epoch=100 --model="pointnet2_part_seg_msg_4k" --tag="STANDING_TREES" --rgbi_channel --batch_size=8 --npoint=4000 --fold_id=0 --sigma="0.8" --num_aug_copies=3 --same_weight=1

# # test standing trees
# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_standing_trees.py --model="pointnet2_part_seg_msg_4k" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-03-30_20-13_STANDING_TREES" --num_point=4000 --rgbi_channel --fold_id=0


# # 30k points
# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_standing_trees.py --epoch=100 --model="pointnet2_part_seg_msg_4k" --tag="STANDING_TREES_30k" --rgbi_channel --batch_size=8 --npoint=30000 --fold_id=0 --sigma="0.8" --num_aug_copies=3 --same_weight=1

# # test standing trees - 30k
# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_standing_trees.py --model="pointnet2_part_seg_msg_4k" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-04-05_12-04_STANDING_TREES_30k" --num_point=30000 --rgbi_channel --fold_id=0


# # 15k points
# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_standing_trees.py --epoch=100 --model="pointnet2_part_seg_msg_4k" --tag="STANDING_TREES_15k" --rgbi_channel --batch_size=8 --npoint=15000 --fold_id=0 --sigma="0.8" --num_aug_copies=3 --same_weight=1

# # test standing trees - 15k
# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_standing_trees.py --model="pointnet2_part_seg_msg_4k" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-04-05_20-50_STANDING_TREES_15k" --num_point=15000 --rgbi_channel --fold_id=0

# test standing trees - 15k - all points input at once - no fps
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_standing_trees_all_pts.py --model="pointnet2_part_seg_msg_4k" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-04-05_20-50_STANDING_TREES_15k" --num_point=15000 --rgbi_channel --fold_id=0

# test standing trees - 30k - all points input at once - no fps
python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/test_standing_trees_all_pts.py --model="pointnet2_part_seg_msg_4k" --batch_size=1 --base_dir="/home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master" --log_dir="2026-04-05_12-04_STANDING_TREES_30k" --num_point=30000 --rgbi_channel --fold_id=0