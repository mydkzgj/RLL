# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import argparse
import os
import sys
import torch

from torch.backends import cudnn

sys.path.append('.')
from config import cfg
from data import make_data_loader
from data import make_data_loader_classification
from engine.trainer import do_train
from modeling import build_model
#from layers import make_loss
from loss import make_G_loss
from loss import make_D_loss

from solver import make_optimizer, WarmupMultiStepLR, make_optimizers


from utils.logger import setup_logger


def train(cfg):
    # prepare dataset
    #train_loader, val_loader, num_query, num_classes = make_data_loader(cfg)
    #CJY at 2019.9.26  利用重新编写的函数处理同仁数据
    train_loader, val_loader, test_loader, classes_list = make_data_loader_classification(cfg)
    num_classes = len(classes_list)

    # build model and load parameter
    model = build_model(cfg, num_classes)
    model.load_param("Base", cfg.TRAIN.TRICK.PRETRAIN_PATH)
    #print(model)

    # loss function
    #loss_func = make_loss(cfg, num_classes)  # modified by gu
    g_loss_func = make_G_loss(cfg, num_classes)
    d_loss_func = make_D_loss(cfg, num_classes)
    loss_funcs = {}
    loss_funcs["G"] = g_loss_func
    loss_funcs["D"] = d_loss_func
    print('Train with the loss type is', "softmax + common") #cfg.LOSS.TYPE)

    # build optimizer
    #optimizer = make_optimizer(cfg, model)
    optimizers = make_optimizers(cfg, model)
    print('Train with the optimizer type is', cfg.SOLVER.OPTIMIZER.NAME)

    # build scheduler （断点续传功能暂时有问题）
    if cfg.SOLVER.SCHEDULER.RETRAIN_FROM_HEAD == True:
        start_epoch = 0
        op_epochs = 10
        schedulers = []
        for epoch_index in range(op_epochs):
            op_schedulers = []
            for i in range(len(optimizers)):
                op_i_scheduler = WarmupMultiStepLR(optimizers[i], cfg.SOLVER.SCHEDULER.STEPS, cfg.SOLVER.SCHEDULER.GAMMA,
                                      cfg.SOLVER.SCHEDULER.WARMUP_FACTOR,
                                      cfg.SOLVER.SCHEDULER.WARMUP_ITERS, cfg.SOLVER.SCHEDULER.WARMUP_METHOD)
                op_schedulers.append(op_i_scheduler)
            schedulers.append(op_schedulers)

    else:
        start_epoch = eval(cfg.TRAIN.TRICK.PRETRAIN_PATH.split('/')[-1].split('.')[0].split('_')[-1])
        scheduler1 = WarmupMultiStepLR(optimizers[0], cfg.SOLVER.SCHEDULER.STEPS, cfg.SOLVER.SCHEDULER.GAMMA,
                                      cfg.SOLVER.SCHEDULER.WARMUP_FACTOR,
                                      cfg.SOLVER.SCHEDULER.WARMUP_ITERS, cfg.SOLVER.SCHEDULER.WARMUP_METHOD,
                                      start_epoch)   # KeyError: "param 'initial_lr' is not specified in param_groups[0] when resuming an optimizer"
        scheduler2 = WarmupMultiStepLR(optimizers[1], cfg.SOLVER.SCHEDULER.STEPS, cfg.SOLVER.SCHEDULER.GAMMA,
                                      cfg.SOLVER.SCHEDULER.WARMUP_FACTOR,
                                      cfg.SOLVER.SCHEDULER.WARMUP_ITERS, cfg.SOLVER.SCHEDULER.WARMUP_METHOD,
                                      start_epoch)  # KeyError: "param 'initial_lr' is not specified in param_groups[0] when resuming an optimizer"



    do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        classes_list,
        optimizers,
        schedulers,      # modify for using self trained model
        loss_funcs,
        start_epoch     # add for using self trained model
    )


def main():
    #解析命令行参数,详见argparse模块
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)    #nargs=argparse.REMAINDER是指所有剩余的参数均转化为一个列表赋值给此项

    args = parser.parse_args()
     
    #os.environ()是python用来获取系统相关信息的。如environ[‘HOME’]就代表了当前这个用户的主目录
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    #此处是指如果有类似yaml重新赋值参数的文件在的话会把它读进来。这也是rbgirshick/yacs模块的优势所在——参数与代码分离
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.TRAIN.DATALOADER.IMS_PER_BATCH = cfg.TRAIN.DATALOADER.CATEGORIES_PER_BATCH * cfg.TRAIN.DATALOADER.INSTANCES_PER_CATEGORY_IN_BATCH
    cfg.VAL.DATALOADER.IMS_PER_BATCH = cfg.VAL.DATALOADER.CATEGORIES_PER_BATCH * cfg.VAL.DATALOADER.INSTANCES_PER_CATEGORY_IN_BATCH
    cfg.TEST.DATALOADER.IMS_PER_BATCH = cfg.TEST.DATALOADER.CATEGORIES_PER_BATCH * cfg.TEST.DATALOADER.INSTANCES_PER_CATEGORY_IN_BATCH
    cfg.freeze()   #最终要freeze一下，prevent further modification，也就是参数设置在这一步就完成了，后面都不能再改变了

    output_dir = cfg.SOLVER.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    #logger主要用于输出运行日志，相比print有一定优势。
    logger = setup_logger("fundus_prediction", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    #哦，此处把config文件又专门读了一遍，并输出了出来
    '''
    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    '''
    logger.info("Running with config:\n{}".format(cfg))


    #？上面的GPU与CUDA是什么关系，这个参数的意义是？
    if cfg.MODEL.DEVICE == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID    # new add by gu
    cudnn.benchmark = True
    train(cfg)


if __name__ == '__main__':
    main()
