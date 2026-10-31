# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_treepartseg_repeat_pts.py --epoch=50

# python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_treepartseg_repeat_pts_v2.py --epoch=300 --model="pointnet_part_seg" --rgbi_channel

python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_treepartseg_all_points.py --epoch=100 --model="pointnet_part_seg" --tag="rgbi_new_sampling" --rgbi_channel

python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_treepartseg_all_points.py --epoch=100 --model="pointnet_part_seg" --tag="xyz_new_sampling"



python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_treepartseg_all_points_orig_pointnet.py --epoch=100 --model="pointnet_part_seg" --tag="xyz_all_points" --batch_size=1

python /home/adhiksa@iu.edu/Research/PointCloud/Models/Pointnet_Pointnet2_pytorch-master/train_treepartseg_all_points_orig_pointnet.py --epoch=100 --model="pointnet_part_seg" --tag="rgbi_all_points" --batch_size=1 --rgbi_channel
