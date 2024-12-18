"""
Function: Flexible FAS metrics: APCER, BPCER, ACER, HTER, EER, AUC, TPR@FPR=10e-4.
Author: Ajian Liu
Date: 2024/12/3
"""
from collections import OrderedDict
import util.utils_FAS as utils
from dassl.evaluation import EvaluatorBase
from dassl.evaluation.build import EVALUATOR_REGISTRY
import torch.nn.functional as F
import numpy as np

@EVALUATOR_REGISTRY.register()
class FAS_CDI(EvaluatorBase):
    """Evaluator for classification."""

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg)
        self._correct_r, self._correct_d, self._correct_i = 0, 0, 0
        self._total_r,   self._total_d,   self._total_i = 0, 0, 0
        self._y_true_r,  self._y_true_d,  self._y_true_i = [], [], []
        self._y_pred_r,  self._y_pred_d,  self._y_pred_i = [], [], []
        self._thr = 'list'  ## grid, list

        self.best_ACER_r, self.best_APCER_r, self.best_BPCER_r, self.best_HTER_r, self.best_AUC_r, self.best_TPR_r = 1.0, 1.0, 1.0, 1.0, 0.0, 0.0
        self.best_ACER_d, self.best_APCER_d, self.best_BPCER_d, self.best_HTER_d, self.best_AUC_d, self.best_TPR_d = 1.0, 1.0, 1.0, 1.0, 0.0, 0.0
        self.best_ACER_i, self.best_APCER_i, self.best_BPCER_i, self.best_HTER_i, self.best_AUC_i, self.best_TPR_i = 1.0, 1.0, 1.0, 1.0, 0.0, 0.0

    def reset(self):
        self._correct_r, self._correct_d, self._correct_i = 0, 0, 0
        self._total_r,   self._total_d,   self._total_i = 0, 0, 0
        self._y_true_r,  self._y_true_d,  self._y_true_i = [], [], []
        self._y_pred_r,  self._y_pred_d,  self._y_pred_i = [], [], []

    def process(self, mo3, gt):
        # mo (torch.Tensor): model output [batch, num_classes]
        # gt (torch.LongTensor): ground truth [batch]

        [logit_r, logit_d, logit_i] = mo3
        pred_r, pred_d, pred_i = logit_r.max(1)[1], logit_d.max(1)[1], logit_i.max(1)[1]
        matc_r, matc_d, matc_i = pred_r.eq(gt).float(), pred_d.eq(gt).float(), pred_i.eq(gt).float()
        self._correct_r += int(matc_r.sum().item())
        self._correct_d += int(matc_d.sum().item())
        self._correct_i += int(matc_i.sum().item())

        self._total_r += gt.shape[0]
        self._total_d += gt.shape[0]
        self._total_i += gt.shape[0]
        prob_r = F.softmax(logit_r, 1)
        prob_d = F.softmax(logit_d, 1)
        prob_i = F.softmax(logit_i, 1)
        for i in range(len(prob_r)):
            score = prob_r[i].data.cpu().numpy()
            self._y_pred_r = np.append(self._y_pred_r, score[1])
            self._y_true_r = np.append(self._y_true_r, gt[i].data.cpu().numpy())
            score = prob_d[i].data.cpu().numpy()
            self._y_pred_d = np.append(self._y_pred_d, score[1])
            self._y_true_d = np.append(self._y_true_d, gt[i].data.cpu().numpy())
            score = prob_i[i].data.cpu().numpy()
            self._y_pred_i = np.append(self._y_pred_i, score[1])
            self._y_true_i = np.append(self._y_true_i, gt[i].data.cpu().numpy())

    def evaluate(self, split="val", threshold=None, eval_only=False):
        results = OrderedDict()
        if split == "val":
            if 'list' in self._thr:
                cur_threshold_r, cur_eer_r, cur_hter_r = utils.get_Threshold_List(self._y_pred_r, self._y_true_r)
                cur_threshold_d, cur_eer_d, cur_hter_d = utils.get_Threshold_List(self._y_pred_d, self._y_true_d)
                cur_threshold_i, cur_eer_i, cur_hter_i = utils.get_Threshold_List(self._y_pred_i, self._y_true_i)
            else:
                cur_threshold_r, cur_eer_r, cur_hter_r = utils.get_Threshold_Grid(self._y_pred_r, self._y_true_r)
                cur_threshold_d, cur_eer_d, cur_hter_d = utils.get_Threshold_Grid(self._y_pred_d, self._y_true_d)
                cur_threshold_i, cur_eer_i, cur_hter_i = utils.get_Threshold_Grid(self._y_pred_i, self._y_true_i)
        elif split == "test":
            [cur_threshold_r, cur_threshold_d, cur_threshold_i] = threshold

            if 'list' in self._thr:
                _, cur_eer_r, cur_hter_r = utils.get_Threshold_List(self._y_pred_r, self._y_true_r)
                _, cur_eer_d, cur_hter_d = utils.get_Threshold_List(self._y_pred_d, self._y_true_d)
                _, cur_eer_i, cur_hter_i = utils.get_Threshold_List(self._y_pred_i, self._y_true_i)
            else:
                _, cur_eer_r, cur_hter_r = utils.get_Threshold_Grid(self._y_pred_r, self._y_true_r)
                _, cur_eer_d, cur_hter_d = utils.get_Threshold_Grid(self._y_pred_d, self._y_true_d)
                _, cur_eer_i, cur_hter_i = utils.get_Threshold_Grid(self._y_pred_i, self._y_true_i)

        cur_apcer_r, cur_bpcer_r, cur_acer_r, cur_acc_r, cur_auc_r, cur_recall2_r, cur_recall3_r, cur_recall4_r = \
            utils.get_Metrics_at_Threshold(self._y_pred_r, self._y_true_r, cur_threshold_r)
        cur_apcer_d, cur_bpcer_d, cur_acer_d, cur_acc_d, cur_auc_d, cur_recall2_d, cur_recall3_d, cur_recall4_d = \
            utils.get_Metrics_at_Threshold(self._y_pred_d, self._y_true_d, cur_threshold_d)
        cur_apcer_i, cur_bpcer_i, cur_acer_i, cur_acc_i, cur_auc_i, cur_recall2_i, cur_recall3_i, cur_recall4_i = \
            utils.get_Metrics_at_Threshold(self._y_pred_i, self._y_true_i, cur_threshold_i)

        # The first (acer) and last (threshold) values will be returned by trainer.test()
        best_Flag = (self.best_ACER_r + self.best_ACER_d + self.best_ACER_i) / 3
        results["Flag"] = (cur_acer_r + cur_acer_d + cur_acer_i) / 3

        results["ACER"] = (cur_acer_r + cur_acer_d + cur_acer_i) / 3
        results["APCER"]=(cur_apcer_r +cur_apcer_d +cur_apcer_i) / 3
        results["BPCER"]=(cur_bpcer_r +cur_bpcer_d +cur_bpcer_i) / 3
        results["HTER"] = (cur_hter_r + cur_hter_d + cur_hter_i) / 3
        results["AUC"] = (cur_auc_r + cur_auc_d + cur_auc_i) / 3
        results["ACC"] = (cur_acc_r + cur_acc_d + cur_acc_i) / 3
        results["EER"] = (cur_eer_r + cur_eer_d + cur_eer_i) / 3
        results["TPR"] = (cur_recall4_r + cur_recall4_r + cur_recall4_r) / 3
        results["Threshold"] = [cur_threshold_r, cur_threshold_d, cur_threshold_i]

        if split == "test":
            is_best = results["Flag"] < best_Flag
            if is_best:
                self.best_APCER_r, self.best_BPCER_r, self.best_ACER_r, self.best_HTER_r, self.best_AUC_r, \
                self.best_ACC_r, self.best_EER_r, self.best_TPR_r = \
                    cur_apcer_r, cur_bpcer_r, cur_acer_r, cur_hter_r, cur_auc_r, cur_acc_r, cur_eer_r, cur_recall4_r

                self.best_APCER_d, self.best_BPCER_d, self.best_ACER_d, self.best_HTER_d, self.best_AUC_d, \
                self.best_ACC_d, self.best_EER_d, self.best_TPR_d = \
                    cur_apcer_d, cur_bpcer_d, cur_acer_d, cur_hter_d, cur_auc_d, cur_acc_d, cur_eer_d, cur_recall4_d

                self.best_APCER_i, self.best_BPCER_i, self.best_ACER_i, self.best_HTER_i, self.best_AUC_i, \
                self.best_ACC_i, self.best_EER_i, self.best_TPR_i = \
                    cur_apcer_i, cur_bpcer_i, cur_acer_i, cur_hter_i, cur_auc_i, cur_acc_i, cur_eer_i, cur_recall4_i
            print(
                "=> best result\n"
                f"*****total_r/d/i: {round(self._total_r, 4):,}/{round(self._total_d, 4):,}/{round(self._total_i, 4):,}\n"
                f"**Accuracy_r/d/i: {round(self.best_ACC_r, 4):,}/{round(self.best_ACC_d, 4):,}/{round(self.best_ACC_i, 4):,}\n"
                f"*****APCER_r/d/i: {round(self.best_APCER_r, 4):,}/{round(self.best_APCER_d, 4):,}/{round(self.best_APCER_i, 4):,}\n"
                f"*****BPCER_r/d/i: {round(self.best_BPCER_r, 4):,}/{round(self.best_BPCER_d, 4):,}/{round(self.best_BPCER_i, 4):,}\n"
                f"******ACER_r/d/i: {round(self.best_ACER_r, 4):,}/{round(self.best_ACER_d, 4):,}/{round(self.best_ACER_i, 4):,}\n"
                f"******HTER_r/d/i: {round(self.best_HTER_r, 4):,}/{round(self.best_HTER_d, 4):,}/{round(self.best_HTER_i, 4):,}\n"
                f"*******AUC_r/d/i: {round(self.best_AUC_r, 4):,}/{round(self.best_AUC_d, 4):,}/{round(self.best_AUC_i, 4):,}\n"
                f"*******TPR_r/d/i: {round(self.best_TPR_r, 4):,}/{round(self.best_TPR_d, 4):,}/{round(self.best_TPR_i, 4):}\n"
                f"*Threshold_r/d/i: {round(cur_threshold_r, 4):,}/{round(cur_threshold_d, 4):,}/{round(cur_threshold_i, 4):,}\n"
                )
        return results


