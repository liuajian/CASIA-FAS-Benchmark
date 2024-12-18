"""
Function: single modal or multi-modal fusion FAS metrics: APCER, BPCER, ACER, HTER, EER, AUC, TPR@FPR=10e-4.
Author: Ajian Liu
Date: 2024/12/3
"""
from collections import OrderedDict
import util.utils_FAS as utils
from dassl.evaluation import EvaluatorBase
from dassl.evaluation.build import EVALUATOR_REGISTRY
import torch.nn.functional as F
import numpy as np

import os
import json

@EVALUATOR_REGISTRY.register()
class FAS(EvaluatorBase):
    """Evaluator for classification."""

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg)
        self._correct = 0
        self._total = 0
        self._y_true = []
        self._y_pred = []
        self._thr = 'grid'  ## grid, list

        self.best_APCER = 1.0
        self.best_BPCER = 1.0
        self.best_ACER = 1.0
        self.best_HTER = 1.0
        self.best_EER = 1.0
        self.best_AUC = 0.0
        self.best_ACC = 0.0
        self.best_TPR = 0.0

    def reset(self):
        self._correct = 0
        self._total = 0
        self._y_true = []
        self._y_pred = []

    def process(self, mo, gt):
        # mo (torch.Tensor): model output [batch, num_classes]
        # gt (torch.LongTensor): ground truth [batch]
        pred = mo.max(1)[1]
        matches = pred.eq(gt).float()
        self._correct += int(matches.sum().item())
        self._total += gt.shape[0]

        prob = F.softmax(mo, 1)
        for i in range(len(prob)):
            score = prob[i].data.cpu().numpy()
            self._y_pred = np.append(self._y_pred, score[1])
            self._y_true = np.append(self._y_true, gt[i].data.cpu().numpy())

    def evaluate(self, split="val", threshold=None):
        results = OrderedDict()
        if split == "val":
            if 'list' in self._thr:
                cur_threshold, cur_eer, cur_hter = utils.get_Threshold_List(self._y_pred, self._y_true)
            else:
                cur_threshold, cur_eer, cur_hter = utils.get_Threshold_Grid(self._y_pred, self._y_true)

        elif split == "test":
            cur_threshold = threshold
            if 'list' in self._thr:
                _, cur_eer, cur_hter = utils.get_Threshold_List(self._y_pred, self._y_true)
            else:
                _, cur_eer, cur_hter = utils.get_Threshold_Grid(self._y_pred, self._y_true)

        cur_apcer, cur_bpcer, cur_acer, cur_acc, cur_auc, cur_recall2, cur_recall3, cur_recall4 = \
            utils.get_Metrics_at_Threshold(self._y_pred, self._y_true, cur_threshold)

        # The first (acer) and last (threshold) values will be returned by trainer.test()
        best_Flag = self.best_ACER
        results["Flag"] = cur_acer
        results["ACER"] = cur_acer
        results["APCER"] = cur_apcer
        results["BPCER"] = cur_bpcer
        results["HTER"] = cur_hter
        results["AUC"] = cur_auc
        results["ACC"] = cur_acc
        results["EER"] = cur_eer
        results["TPR"] = cur_recall4
        results["Threshold"] = cur_threshold

        if split == "test":
            is_best = results["Flag"] < best_Flag
            if is_best:
                print('update:', results["Flag"])
                self.best_APCER, self.best_BPCER, self.best_ACER, self.best_HTER, self.best_AUC, \
                self.best_ACC, self.best_EER, self.best_TPR = \
                cur_apcer, cur_bpcer, cur_acer, cur_hter, cur_auc, cur_acc, cur_eer, cur_recall4
            print(
                "=> best result\n"
                f"* total: {self._total:,}\n"
                f"* ACC: {self.best_ACC:,}\n"
                f"* APCER: {self.best_APCER:,}\n"
                f"* BPCER: {self.best_BPCER:,}\n"
                f"* ACER: {self.best_ACER:,}\n"
                f"* HTER: {self.best_HTER:,}\n"
                f"* AUC: {self.best_AUC:,}\n"
                f"* EER: {self.best_EER:,}\n"
                f"* TPR: {self.best_TPR:}\n"
                f"* Threshold: {cur_threshold:,}\n"
                )
        return results

