import numpy as np
import os
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable
import utils.transforms as trans
import utils.utils as util
import utils.metric as mc
import cv2

import cfg.CDD as cfg
import dataset.CDD as dates
from train import check_dir, various_distance

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def single_layer_similar_heatmap(output_t0, output_t1, dist_flag):
    interp = nn.Upsample(size=[cfg.TRANSFROM_SCALES[1],cfg.TRANSFROM_SCALES[0]], mode='bilinear', align_corners=True)
    n, c, h, w = output_t0.data.shape
    out_t0_rz = torch.transpose(output_t0.view(c, h * w), 1, 0)
    out_t1_rz = torch.transpose(output_t1.view(c, h * w), 1, 0)
    distance = various_distance(out_t0_rz, out_t1_rz, dist_flag=dist_flag)
    similar_distance_map = distance.view(h, w).data.cpu().numpy()
    similar_distance_map_rz = interp(Variable(torch.from_numpy(similar_distance_map[np.newaxis, np.newaxis, :])))
    return similar_distance_map_rz.data.cpu().numpy()


# def calculate_f_score(metric, threshold):
#     # Calculate f_score based on the training best threshold
#     index = round(threshold * 255)
#     fn = metric['total_fn'][index]
#     fp = metric['total_fp'][index]
#     tp = metric['total_posnum'] - fn
#     tn = metric['total_negnum'] - fp
#     test_recall = tp / float(metric['total_posnum'])
#     test_precision = tp / (tp + fp + 1e-10)
#     beta = cfg.beta
#     betasq = beta ** 2
#     test_f_score = (1 + betasq) * (test_precision * test_recall) / ((betasq * test_precision) + test_recall + 1e-10)
#     return test_f_score


def test(net, test_dataloader, save_test_dir, save_roc_dir):
    net.eval()
    cont_conv5_total,cont_fc_total,cont_embedding_total,num = 0.0,0.0,0.0,0.0
    metric_for_conditions = util.init_metric_for_class_for_cmu(1)
    
    # Run inference for test images set, get prob_changes and calculate FN and FP for threshol array:
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dataloader):
            inputs1,input2, targets, filename, height, width = batch
            height, width, filename = height.numpy()[0], width.numpy()[0], filename[0]
            inputs1,input2,targets = inputs1.cuda(),input2.cuda(), targets.cuda()
            inputs1,inputs2,targets = Variable(inputs1, volatile=True),Variable(input2,volatile=True) ,Variable(targets)
            targets = torch.round(targets.float() / 255).float()
            out_conv5,out_fc,out_embedding = net(inputs1,inputs2)
            out_embedding_t0,out_embedding_t1 = out_embedding
            embedding_distance_map = single_layer_similar_heatmap(out_embedding_t0, out_embedding_t1, 'l2')
            cont_embedding = mc.RMS_Contrast(embedding_distance_map)
            cont_embedding_total += cont_embedding
            num += 1
            prob_change = embedding_distance_map[0][0]
        
            gt = targets.data.cpu().numpy()
            FN, FP, posNum, negNum = mc.eval_image_rewrite(gt[0], prob_change, cl_index=1)
            metric_for_conditions[0]['total_fp'] += FP
            metric_for_conditions[0]['total_fn'] += FN
            metric_for_conditions[0]['total_posnum'] += posNum
            metric_for_conditions[0]['total_negnum'] += negNum
            cont_embedding_mean = cont_embedding_total / num
            changes = cv2.threshold(prob_change * 256, 255, 255, cv2.THRESH_BINARY)[1]
            changes_filename = os.path.join(save_test_dir, filename)
            cv2.imwrite(changes_filename, changes)

    # Calculate f_score array
    thresh = np.array(range(0, 256)) / 255.0
    conds = metric_for_conditions.keys()
    for cond_name in conds:
        total_posnum = metric_for_conditions[cond_name]['total_posnum']
        total_negnum = metric_for_conditions[cond_name]['total_negnum']
        total_fn = metric_for_conditions[cond_name]['total_fn']
        total_fp = metric_for_conditions[cond_name]['total_fp']
        metric_dict = mc.pxEval_maximizeFMeasure(total_posnum, total_negnum,
                                                total_fn, total_fp, thresh=thresh)
        metric_for_conditions[cond_name].setdefault('metric', metric_dict)
        metric_for_conditions[cond_name].setdefault('contrast_embedding',cont_embedding_mean)

    # test_f_score = calculate_f_score(metric_for_conditions[0], BestThresh)

    f_score_total = 0.0
    for cond_name in conds:
        pr, recall, f_score = metric_for_conditions[cond_name]['metric']['precision'], \
            metric_for_conditions[cond_name]['metric']['recall'], \
                metric_for_conditions[cond_name]['metric']['MaxF']
        roc_save_dir = os.path.join(save_roc_dir, 'test')
        check_dir(roc_save_dir)
        mc.save_metric2disk(metric_for_conditions, roc_save_dir)
        roc_save_filepath = os.path.join(roc_save_dir, 'test_roc.png')
        mc.plotPrecisionRecall(pr, recall, roc_save_filepath, benchmark_pr=None)
        f_score_total += f_score

    print("mean f_score={}".format(f_score_total/(len(conds))))
    return f_score_total / len(conds)

def main():
    ##### configs #####
    ##### load datasets #####
    test_transform_det = trans.Compose([trans.Scale(cfg.TRANSFROM_SCALES), ])
    test_data = dates.Dataset(cfg.TEST_DATA_PATH, cfg.TEST_LABEL_PATH, 
                            cfg.TEST_TXT_PATH, 'test', transform=True, 
                            transform_med=test_transform_det)
    test_loader = Data.DataLoader(test_data, batch_size=cfg.BATCH_SIZE, 
                                    shuffle=False, num_workers=4, pin_memory=True)

    ##### build models #####
    import model.DASNET as models
    model = models.SiameseNet(norm_flag='l2')
    checkpoint = torch.load(cfg.TRAINED_BEST_PERFORMANCE_CKPT)
    model.load_state_dict(checkpoint['state_dict'])
    print('load ckpt success')

    model = model.cuda()
    save_test_dir = os.path.join(cfg.SAVE_PRED_PATH, 'test_imgs')
    save_roc_dir = os.path.join(cfg.SAVE_PRED_PATH, 'roc')
    check_dir(save_test_dir), check_dir(save_roc_dir)
    #######
    model.eval()
    test(model, test_loader, save_test_dir, save_roc_dir)

  
if __name__ == '__main__':
   main()