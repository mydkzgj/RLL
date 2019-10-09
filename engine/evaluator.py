# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
import logging

import torch
import torch.nn as nn
from ignite.engine import Engine, Events

from ignite.metrics import Accuracy
from ignite.metrics import Precision
from ignite.metrics import Recall
from ignite.metrics import ConfusionMatrix

from sklearn.metrics import roc_curve, auc
from utils.plot_roc import plotROC_OneClass, plotROC_MultiClass
import numpy as np


def convert_to_one_hot(y, C):
    return np.eye(C)[y.reshape(-1)]

def create_supervised_evaluator(model, metrics, loss_fn, device=None):
    """
    Factory function for creating an evaluator for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        metrics (dict of str - :class:`ignite.metrics.Metric`): a map of metric names to Metrics
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
    Returns:
        Engine: an evaluator engine with supervised inference function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            data, labels = batch
            data = data.to(device) if torch.cuda.device_count() >= 1 else data
            #注：此处labels也要放入cuda，才能做下面的acc计算
            labels = labels.to(device) if torch.cuda.device_count() >= 1 else labels
            scores, feats = model(data)
            #pre_labels = score.max(1)[1]
            #acc = (scores.max(1)[1] == labels).float().mean()
            loss = loss_fn(scores, feats, labels)
            return {"scores":scores, "labels":labels, "feat":feats, "loss":loss.item()}

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine

def do_inference(
        cfg,
        model,
        test_loader,
        num_classes,
        loss_fn,
        plotROC = False
):
    device = cfg.MODEL.DEVICE

    logger = logging.getLogger("fundus_prediction.inference")
    logger.info("Enter inferencing")
    metrics_eval = {"overall_accuracy": Accuracy(output_transform=lambda x: (x["scores"], x["labels"])),
                    "precision": Precision(output_transform=lambda x: (x["scores"], x["labels"])),
                    "recall": Recall(output_transform=lambda x: (x["scores"], x["labels"])),
                    "confusion_matrix": ConfusionMatrix(num_classes=num_classes, output_transform=lambda x: (x["scores"], x["labels"])),
                    }
    evaluator = create_supervised_evaluator(model, metrics=metrics_eval, loss_fn=loss_fn, device=device)

    if plotROC == True:
        y_pred = []
        y_label = []
        @evaluator.on(Events.ITERATION_COMPLETED, y_pred, y_label)
        def combineTensor(engine, y_pred, y_label):
            scores = engine.state.output["scores"].cpu().numpy().tolist()
            labels = engine.state.output["labels"].cpu().numpy().tolist()
            y_pred = y_pred.extend(scores)   #注意，此处要用extend，否则+会产生新列表
            y_label = y_label.extend(labels)

    @evaluator.on(Events.EPOCH_COMPLETED)
    def log_inference_results(engine):
        precision = engine.state.metrics['precision'].numpy().tolist()
        avg_accuracy = 0
        for index, ap in enumerate(precision):
            avg_accuracy = avg_accuracy + ap
            precision[index] = float("{:.2f}".format(ap))
        avg_accuracy = avg_accuracy / len(precision)

        recall = engine.state.metrics['recall'].numpy().tolist()
        avg_recall = 0
        for index, ar in enumerate(recall):
            avg_recall = avg_recall + ar
            recall[index] = float("{:.2f}".format(ar))
        avg_recall = avg_recall / len(recall)

        confusion_matrix = engine.state.metrics['confusion_matrix'].numpy()

        overall_accuracy = engine.state.metrics['overall_accuracy']
        logger.info("Test Results")
        logger.info("Precision: {}, Average_Accuracy: {:.2f}".format(precision, avg_accuracy))
        logger.info("Recall: {}, Average_Recall: {:.2f}".format(recall, avg_recall))
        logger.info("Overall_Accuracy: {:.3f}".format(overall_accuracy))
        logger.info("ConfusionMatrix: x-groundTruth  y-predict \n {}".format(confusion_matrix))

    evaluator.run(test_loader)

    if plotROC == True:
        # Plot ROC
        # convert List to numpy
        y_label = np.array(y_label)
        y_label = convert_to_one_hot(y_label, num_classes)
        y_pred = np.array(y_pred)

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_label[:, i], y_pred[:, i], pos_label=i)
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_label.ravel(), y_pred.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        plotROC_MultiClass(fpr, tpr, roc_auc, num_classes)

