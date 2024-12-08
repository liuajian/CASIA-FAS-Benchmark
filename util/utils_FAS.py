"""
Function: Utils for Face Anti-spoofing
Author: Ajian Liu
Date: 2024.12.2
"""
import os, torch, sys, math, datetime
from six import iteritems
from sklearn.metrics import roc_curve, auc
from scipy import interp
import numpy as np
from torchvision import utils
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F

###############################################################
# Utils for Common
###############################################################
def check_if_exist(path):
    """function: Determine if the file exists"""
    return os.path.exists(path)

def make_if_not_exist(path):
    """function: Determine if the file exists, and make"""
    if not os.path.exists(path):
        os.makedirs(path)

def set_env(args):
    # os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
    assert -1 not in gpu_ids  ### Temporarily not allowed cpu
    if len(gpu_ids) > 0:torch.cuda.set_device(gpu_ids[0])
    use_cuda = torch.cuda.is_available()
    if use_cuda and gpu_ids[0] >= 0:device = torch.device('cuda:%d' % gpu_ids[0])
    else:device = torch.device('cpu')
    return gpu_ids, use_cuda, device

def mk_jobs(args):
    ### subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    folders = []
    for folder in ['logs', 'models', 'outputs', 'scores']:
        folder = os.path.join(args.job_out, folder, args.protocol, args.subdir)
        make_if_not_exist(folder)
        folders.append(folder)
    write_args_to_file(args, os.path.join(folders[3], 'result.txt'))
    print('mkdir: logs, models, outputs, scores, result.txt')
    return folders[0], folders[1], folders[2], folders[3]

def write_args_to_file(args, filename):
    """
    :param args:
    :param filename:
    :return: write args parameter to file
    """
    with open(filename, 'a') as f:
        for key, value in iteritems(vars(args)):
            f.write('%s: %s\n' % (key, str(value)))

def get_eta_time(num_batches, batch_idx, epoch, max_epochs, batch_time):
    nb_remain = 0
    nb_remain += num_batches - batch_idx - 1
    nb_remain += (max_epochs - epoch - 1) * num_batches
    eta_seconds = batch_time.avg * nb_remain
    eta = str(datetime.timedelta(seconds=int(eta_seconds)))
    return eta

def convert_secs2time(epoch_time):
    need_hour = int(epoch_time / 3600)
    need_mins = int((epoch_time - 3600*need_hour) / 60)
    need_secs = int(epoch_time - 3600*need_hour - 60*need_mins)
    return need_hour, need_mins, need_secs

def time_to_str(t, mode='min'):
    if mode == 'min':
        t = int(t) / 60
        hr = t // 60
        min = t % 60
        return '%2d hr %02d min'%(hr, min)
    elif mode == 'sec':
        t = int(t)
        min = t // 60
        sec = t % 60
        return '%2d min %02d sec'%(min, sec)
    else:
        raise NotImplementedError

def save_image(tensor, outdir, epoch_iter, prefix='XY', nrow=4):
    filename = os.path.join(outdir, '{}_iter_{}.jpg'.format(str(epoch_iter), prefix))
    utils.save_image(tensor, filename, nrow=int(nrow), normalize=True)

###############################################################
# Optimizer and Lr
###############################################################
def adjust_lr(args, optimizer, epoch, lr_decay_epochs, lr_decay=0.1):
    lr = args.lr
    if args.warm_cosine[1]:
        eta_min = lr * (lr_decay ** 4)
        lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / args.max_epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)
        # for s in lr_decay_epochs:
        #     if epoch >= s:
        #         lr *= lr_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def warmup_learning_rate(args, lr_c, batch_id, epoch, total_batches, optimizer):
    # warm-up for large-batch training
    warm_epochs = 1
    if args.warm_cosine[1]:
        eta_min = args.lr * (args.lr_decay_rate ** 3)
        warmup_to = eta_min + \
                    (args.lr - eta_min) * \
                    (1 + math.cos(math.pi * warm_epochs / args.max_epochs)) / 2
    else:
        warmup_to = args.lr

    ### update lr
    warmup_from = 0.01
    if args.batch_size > 256:warm = True
    else:warm = False
    if warm and epoch <= warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / (warm_epochs * total_batches)
        lr = warmup_from + p * (warmup_to - warmup_from)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            print('update lr by warm-up', 'epoch:{}, batch_id:{}, lr:{}'.format(epoch, batch_id, lr))
        return lr
    else:
        return lr_c

def cross_entropy(logit, truth):
    prob = F.softmax(logit, 1)
    value, top = prob.topk(1, dim=1, largest=True, sorted=True)
    correct = top.eq(truth.view(-1, 1).expand_as(top))
    correct = correct.data.cpu().numpy()
    correct = np.mean(correct)
    return correct, prob

###############################################################
# Test and Metric:
###############################################################
def eval_state(probs, labels, thr):
    predict = probs >= thr
    TP = np.sum((labels == 1) & (predict == True))
    FN = np.sum((labels == 1) & (predict == False))
    TN = np.sum((labels == 0) & (predict == False))
    FP = np.sum((labels == 0) & (predict == True))
    return TN, FN, FP, TP

def get_Threshold_List(probs, labels):
    fpr, tpr, thresholds = roc_curve(labels, probs, pos_label=1)
    fnr = 1 - tpr
    right_index = np.argmin(np.abs(fnr - fpr))
    best_thr = thresholds[right_index]
    ##########################################
    EER = fpr[right_index]

    FRR = 1 - tpr
    HTER = ((fpr + FRR) / 2.0)[right_index]
    return best_thr, EER, HTER

def get_Threshold_Grid(probs, labels, grid_density=10000):
    def get_threshold(grid_density):
        thresholds = []
        for i in range(grid_density + 1):
            thresholds.append(0.0 + i * 1.0 / float(grid_density))
        thresholds.append(1.1)
        return thresholds
    thresholds = get_threshold(grid_density)
    min_dist = 1.0
    min_dist_states = []
    FRR_list = []
    FAR_list = []
    for thr in thresholds:
        TN, FN, FP, TP = eval_state(probs, labels, thr)
        if (FN + TP == 0):
            FRR = TPR = 1.0
            FAR = FP / float(FP + TN)
            TNR = TN / float(TN + FP)
        elif (FP + TN == 0):
            TNR = FAR = 1.0
            FRR = FN / float(FN + TP)
            TPR = TP / float(TP + FN)
        else:
            FAR = FP / float(FP + TN)
            FRR = FN / float(FN + TP)
            TNR = TN / float(TN + FP)
            TPR = TP / float(TP + FN)
        dist = math.fabs(FRR - FAR)
        FAR_list.append(FAR)
        FRR_list.append(FRR)
        if dist <= min_dist:
            min_dist = dist
            min_dist_states = [FAR, FRR, thr]
    thr = min_dist_states[2]
    #############################
    EER = (min_dist_states[0] + min_dist_states[1]) / 2.0
    HTER = EER
    return thr, EER, HTER

def get_Metrics_at_Threshold(probs, labels, thr):
    TN, FN, FP, TP = eval_state(probs, labels, thr)
    FAR = FP / (FP + TN)            ### False Acceptance Rate
    FRR = FN / (TP + FN)            ### False Rejection Rate
    HTER = (FAR + FRR) / 2          ### Half Total Error Rate
    APCER = FRR                     ### Normal Presentation Classification Error Rate
    NPCER = FAR                     ### Attack Presentation Classification Error Rate == FPR
    ACER = (APCER + NPCER) / 2      ### Average Classification Error Rate
    FPR = FAR
    TPR = 1 - FRR
    ACC = (TP + TN) / (TP + FP + FN + TN)  ### classification accuracy
    AUC = roc_auc_score(labels, probs)

    test_fpr, test_tpr, _ = roc_curve(labels, probs, pos_label=1)
    Recall2 = interp(0.01, test_fpr, test_tpr)
    Recall3 = interp(0.001, test_fpr, test_tpr)
    Recall4 = interp(0.0001, test_fpr, test_tpr)
    return APCER, NPCER, ACER, ACC, AUC, Recall2, Recall3, Recall4




