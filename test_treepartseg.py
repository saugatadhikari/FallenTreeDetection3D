"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
from data_utils.ShapeNetDataLoader import TreePartNormalDataset
import torch
import logging
import sys
import importlib
from tqdm import tqdm
import numpy as np
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

seg_classes = {'standing_trees': [0], 'fallen_trees': [1, 2, 3]}
PARTS_LABELS = [0, 1, 2, 3]
LABELS_TO_PARTS = {0: 'standing_tree', 1: 'fallen_trunk', 2: 'root_ball', 3: 'pit'}

seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in testing')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=2048, help='point Number')
    parser.add_argument('--log_dir', type=str, required=True, help='experiment root')
    parser.add_argument('--base_dir', type=str, required=True, help='working dir')
    parser.add_argument('--normal', action='store_true', default=False, help='use normals')
    parser.add_argument('--num_votes', type=int, default=3, help='aggregate segmentation scores with voting')
    parser.add_argument('--model', type=str, required=True, help='model name')
    parser.add_argument('--rgbi_channel', action='store_true', default=False, help='use RGBI channels')
    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = args.base_dir + '/tree_log/' + args.model + "/" + args.log_dir
    
    predictions_dir = os.path.join(experiment_dir, 'predictions/')
    os.makedirs(predictions_dir, exist_ok=True)
    
    points_dir = os.path.join(experiment_dir, 'points/')
    os.makedirs(points_dir, exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    root = '/home/adhiksa@iu.edu/Research/PointCloud/Data/TreePartAnnotation/data/folders/'

    TEST_DATASET = TreePartNormalDataset(root=root, npoints=args.num_point, split='test', normal_channel=args.normal, use_rgbi=args.rgbi_channel)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)
    log_string("The number of test data is: %d" % len(TEST_DATASET))
    num_classes = 2
    num_part = 4

    '''MODEL LOADING'''
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(num_part, num_classes, normal_channel=args.normal, use_rgbi=args.rgbi_channel).cuda()
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])

    with torch.no_grad():
        test_metrics = {}
        total_correct = 0
        total_seen = 0
        total_seen_class = [0 for _ in range(num_part)]
        total_correct_class = [0 for _ in range(num_part)]
        shape_ious = {cat: [] for cat in seg_classes.keys()}
        seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
        
        
        part_ious_dict = {label: [] for label in PARTS_LABELS}
        
        part_precision_dict = {label: [] for label in PARTS_LABELS}
        part_recall_dict = {label: [] for label in PARTS_LABELS}
        part_f1_dict = {label: [] for label in PARTS_LABELS}

        for cat in seg_classes.keys():
            for label in seg_classes[cat]:
                seg_label_to_cat[label] = cat

        classifier = classifier.eval()
        for batch_id, (points, label, target, data_chosen, filename) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
            batchsize, num_point, _ = points.size()
            cur_batch_size, NUM_POINT, _ = points.size()
            points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
            points = points.transpose(2, 1)
            vote_pool = torch.zeros(target.size()[0], target.size()[1], num_part).cuda()

            for _ in range(args.num_votes):
                seg_pred, _ = classifier(points, to_categorical(label, num_classes))
                vote_pool += seg_pred

            seg_pred = vote_pool / args.num_votes
            cur_pred_val = seg_pred.cpu().data.numpy()
            cur_pred_val_logits = cur_pred_val
            cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
            target = target.cpu().data.numpy()

            for i in range(cur_batch_size):
                cat = seg_label_to_cat[target[i, 0]]
                logits = cur_pred_val_logits[i, :, :]
                cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]
                
            correct = np.sum(cur_pred_val == target)
            total_correct += correct
            total_seen += (cur_batch_size * NUM_POINT)

            for l in range(num_part):
                total_seen_class[l] += np.sum(target == l)
                total_correct_class[l] += (np.sum((cur_pred_val == l) & (target == l)))

            for i in range(cur_batch_size):
                segp = cur_pred_val[i, :]
                segl = target[i, :]

                cat = seg_label_to_cat[segl[0]]
                part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
                
                for l in seg_classes[cat]:
                    # Get masks
                    pred_mask = (segp == l)
                    true_mask = (segl == l)

                    # Counts
                    TP = np.sum(pred_mask & true_mask)
                    FP = np.sum(pred_mask & ~true_mask)
                    FN = np.sum(~pred_mask & true_mask)

                    # IoU
                    if (np.sum(true_mask) == 0) and (np.sum(pred_mask) == 0):
                        part_ious[l - seg_classes[cat][0]] = 1.0
                        part_ious_dict[l].append(1.0)

                        # Edge case: no prediction, no ground truth => perfect score
                        part_precision_dict[l].append(1.0)
                        part_recall_dict[l].append(1.0)
                        part_f1_dict[l].append(1.0)
                    else:
                        iou = TP / float(TP + FP + FN) if (TP + FP + FN) > 0 else 0.0
                        precision = TP / float(TP + FP) if (TP + FP) > 0 else 0.0
                        recall = TP / float(TP + FN) if (TP + FN) > 0 else 0.0
                        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

                        part_ious[l - seg_classes[cat][0]] = iou
                        part_ious_dict[l].append(iou)
                        part_precision_dict[l].append(precision)
                        part_recall_dict[l].append(recall)
                        part_f1_dict[l].append(f1)
                    
                    
                shape_ious[cat].append(np.mean(part_ious))
                
#                 save prediction as numpy
                test_filename = os.path.basename(filename[i])
                pred_filepath = os.path.join(predictions_dir, test_filename)
                points_filepath = os.path.join(points_dir, test_filename)
                
                np.save(pred_filepath, cur_pred_val[i])
                np.save(points_filepath, data_chosen[i].cpu())

            correct = np.sum(cur_pred_val == target)
            total_correct += correct
            total_seen += (cur_batch_size * NUM_POINT)

            for l in range(num_part):
                total_seen_class[l] += np.sum(target == l)
                total_correct_class[l] += (np.sum((cur_pred_val == l) & (target == l)))

            for i in range(cur_batch_size):
                segp = cur_pred_val[i, :]
                segl = target[i, :]
                cat = seg_label_to_cat[segl[0]]
                part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
                for l in seg_classes[cat]:
                    if (np.sum(segl == l) == 0) and (
                            np.sum(segp == l) == 0):  # part is not present, no prediction as well
                        part_ious[l - seg_classes[cat][0]] = 1.0
                    else:
                        part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                            np.sum((segl == l) | (segp == l)))
                shape_ious[cat].append(np.mean(part_ious))
                
                # save prediction as numpy
                test_filename = os.path.basename(filename[i])
                pred_filepath = os.path.join(predictions_dir, test_filename)
                points_filepath = os.path.join(points_dir, test_filename)
                
                np.save(pred_filepath, cur_pred_val[i])
                np.save(points_filepath, data_chosen[i].cpu())


                

        all_shape_ious = []
        for cat in shape_ious.keys():
            for iou in shape_ious[cat]:
                all_shape_ious.append(iou)
            shape_ious[cat] = np.mean(shape_ious[cat])
        mean_shape_ious = np.mean(list(shape_ious.values()))
        test_metrics['accuracy'] = total_correct / float(total_seen)
        test_metrics['class_avg_accuracy'] = np.mean(
            np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float64))
        for cat in sorted(shape_ious.keys()):
            log_string('eval mIoU of %s %f' % (cat + ' ' * (14 - len(cat)), shape_ious[cat]))
            
        for label in PARTS_LABELS:
            iou = np.mean(part_ious_dict[label])
            precision = np.mean(part_precision_dict[label])
            recall = np.mean(part_recall_dict[label])
            f1 = np.mean(part_f1_dict[label])
            
            log_string(f'eval IoU of {LABELS_TO_PARTS[label]} is: {iou}')
            log_string('----------------------------------------------------')
            log_string(f'eval Precision of {LABELS_TO_PARTS[label]} is: {precision}')
            log_string(f'eval Recall of {LABELS_TO_PARTS[label]} is: {recall}')
            log_string(f'eval F1 Score of {LABELS_TO_PARTS[label]} is: {f1}')
            log_string('----------------------------------------------------')
        test_metrics['class_avg_iou'] = mean_shape_ious
        test_metrics['instance_avg_iou'] = np.mean(all_shape_ious)

    log_string('Accuracy is: %.5f' % test_metrics['accuracy'])
    log_string('Class avg accuracy is: %.5f' % test_metrics['class_avg_accuracy'])
    log_string('Class avg mIOU is: %.5f' % test_metrics['class_avg_iou'])
    log_string('Instance avg mIOU is: %.5f' % test_metrics['instance_avg_iou'])


if __name__ == '__main__':
    args = parse_args()
    main(args)
