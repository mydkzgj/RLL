# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import argparse
import os
import sys
from os import mkdir

import torch
from torch.backends import cudnn

sys.path.append('.')
from config import cfg
#from data import make_data_loader
from data import make_data_loader_classification
from engine.evaluator import do_inference
from modeling import build_model
from utils.logger import setup_logger

from layers import make_loss

from torch.utils.tensorboard import SummaryWriter



def main():
    parser = argparse.ArgumentParser(description="ReID Baseline Inference")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.TRAIN.DATALOADER.IMS_PER_BATCH = cfg.TRAIN.DATALOADER.CATEGORIES_PER_BATCH * cfg.TRAIN.DATALOADER.INSTANCES_PER_CATEGORY_IN_BATCH
    cfg.VAL.DATALOADER.IMS_PER_BATCH = cfg.VAL.DATALOADER.CATEGORIES_PER_BATCH * cfg.VAL.DATALOADER.INSTANCES_PER_CATEGORY_IN_BATCH
    cfg.TEST.DATALOADER.IMS_PER_BATCH = cfg.TEST.DATALOADER.CATEGORIES_PER_BATCH * cfg.TEST.DATALOADER.INSTANCES_PER_CATEGORY_IN_BATCH
    cfg.freeze()

    output_dir = cfg.SOLVER.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        mkdir(output_dir)

    logger = setup_logger("fundus_prediction", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DEVICE == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    cudnn.benchmark = True

    #build
    train_loader, val_loader, test_loader, classes_list = make_data_loader_classification(cfg)
    num_classes = len(classes_list)

    model = build_model(cfg, num_classes)
    model.load_param("Overall", cfg.TEST.WEIGHT)

    loss_fn = make_loss(cfg, num_classes)  # modified by gu

    writer_test = SummaryWriter(cfg.SOLVER.OUTPUT_DIR + "/summary/test")

    model_save_epoch = cfg.TEST.WEIGHT.split('/')[-1].split('.')[0].split('_')[-1]
    a= len(train_loader)
    if model_save_epoch.isdigit() == True:
        step = len(train_loader) * int(model_save_epoch)
    else:
        step = 0

    metrics = do_inference(cfg, model, test_loader, classes_list, loss_fn, plotFlag=True)

    for preKey in metrics['precision'].keys():
        writer_test.add_scalar("Precision/" + str(preKey), metrics['precision'][preKey], step)

    for recKey in metrics['recall'].keys():
        writer_test.add_scalar("Recall/" + str(recKey), metrics['recall'][recKey], step)

    for aucKey in metrics['roc_auc'].keys():
        writer_test.add_scalar("ROC_AUC/" + str(aucKey), metrics['roc_auc'][aucKey], step)

    writer_test.add_scalar("OverallAccuracy", metrics["overall_accuracy"], step)

    # writer.add_scalar("Val/"+"confusion_matrix", metrics['confusion_matrix'], step)

    # 混淆矩阵 和 ROC曲线可以用图的方式来存储
    roc_numpy = metrics["roc_figure"]
    writer_test.add_image("ROC", roc_numpy, step, dataformats='HWC')

    confusion_matrix_numpy = metrics["confusion_matrix_numpy"]
    writer_test.add_image("ConfusionMatrix", confusion_matrix_numpy, step, dataformats='HWC')

    writer_test.close()


if __name__ == '__main__':
    main()
