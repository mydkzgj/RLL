# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import logging

import torch
import torch.nn as nn
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer

from ignite.metrics import RunningAverage
from ignite.metrics import Accuracy
from ignite.metrics import Precision

from engine.evaluator import do_inference


#CJY at 2019.9.24 既然这里面定义的与inference.py 中的一样能不能直接引用
#还没加

global ITER
ITER = 0

def create_supervised_trainer(model, optimizer, metrics, loss_fn, device=None):
    """
    Factory function for creating a trainer for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        optimizer (`torch.optim.Optimizer`): the optimizer to use
        loss_fn (torch.nn loss function): the loss function to use
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.

    Returns:
        Engine: a trainer engine with supervised update function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

        #optimizer加载进来的是cpu类型，需要手动转成gpu。
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        imgs, labels = batch   #这个格式应该跟collate_fn的处理方式对应
        imgs = imgs.to(device) if torch.cuda.device_count() >= 1 else imgs
        labels = labels.to(device) if torch.cuda.device_count() >= 1 else labels
        scores, feats = model(imgs)
        loss = loss_fn(scores, feats, labels)
        loss.backward()
        optimizer.step()
        # compute acc
        #acc = (scores.max(1)[1] == labels).float().mean()
        return {"scores":scores, "labels":labels, "loss":loss.item()}

    engine = Engine(_update)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        num_classes,
        optimizer,
        scheduler,
        loss_fn,
        start_epoch
):
    #1.先把cfg中的参数导出
    epochs = cfg.SOLVER.MAX_EPOCHS
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    output_dir = cfg.SOLVER.OUTPUT_DIR
    device = cfg.MODEL.DEVICE


    #2.构建模块
    logger = logging.getLogger("fundus_prediction.train")
    logger.info("Start training")
    """"
    metrics_train = {"avg_loss":  RunningAverage(output_transform=lambda x: x["loss"]),
                    "avg_precision": RunningAverage(output_transform=lambda x: x["precision"]),
                    "avg_accuracy": RunningAverage(output_transform=lambda x: x["accuracy"])}
    """
    metrics_train = {"avg_loss": RunningAverage(output_transform=lambda x: x["loss"]),
                     "avg_precision": RunningAverage(Precision(output_transform=lambda x: (x["scores"], x["labels"]))),
                     "avg_accuracy": RunningAverage(Accuracy(output_transform=lambda x: (x["scores"], x["labels"])))}  #由于训练集样本均衡后远离原始样本集，故只采用平均metric
    trainer = create_supervised_trainer(model, optimizer, metrics_train, loss_fn, device=device)

    #CJY  at 2019.9.26
    def output_transform(output):
        # `output` variable is returned by above `process_function`
        y_pred = output['scores']
        y = output['labels']
        return y_pred, y  # output format is according to `Accuracy` docs

    metrics_eval = {"overall_accuracy": Accuracy(output_transform=output_transform),
                    "precision": Precision(output_transform=output_transform)}

    checkpointer = ModelCheckpoint(output_dir, cfg.MODEL.NAME, checkpoint_period, n_saved=10, require_empty=False)
    timer = Timer(average=True)

    #3.将模块与engine联系起来attach
    #CJY at 2019.9.23
    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model': model,
                                                                     'optimizer': optimizer})
    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)


    #4.事件处理函数
    @trainer.on(Events.STARTED)
    def start_training(engine):
        engine.state.epoch = start_epoch
        do_inference(cfg, model, val_loader, num_classes, loss_fn)

    @trainer.on(Events.EPOCH_STARTED)
    def adjust_learning_rate(engine):
        scheduler.step()

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        global ITER
        ITER += 1

        if ITER % log_period == 0:
            avg_precision = engine.state.metrics['avg_precision'].numpy().tolist()
            for index, ap in enumerate(avg_precision):
                avg_precision[index] = float("{:.2f}".format(ap))
            logger.info("Epoch[{}] Iteration[{}/{}] Avg_Loss: {:.3f}, Avg_Pre: {}, Avg_Acc: {:.3f}, Base Lr: {:.2e}"
                        .format(engine.state.epoch, ITER, len(train_loader),
                                engine.state.metrics['avg_loss'], avg_precision, engine.state.metrics['avg_accuracy'],
                                scheduler.get_lr()[0]))
        if len(train_loader) == ITER:
            ITER = 0

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                    .format(engine.state.epoch, timer.value() * timer.step_count,
                            train_loader.batch_size / timer.value()))
        logger.info('-' * 10)
        timer.reset()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        if engine.state.epoch % eval_period == 0:
            do_inference(cfg, model, val_loader, num_classes, loss_fn)

            """
            evaluator.run(val_loader)
            precision = evaluator.state.metrics['precision'].numpy().tolist()
            avg_accuracy = 0
            for index, ap in enumerate(precision):
                avg_accuracy = avg_accuracy + ap
                precision[index] = float("{:.2f}".format(ap))
            avg_accuracy = avg_accuracy / len(precision)
            overall_accuracy = evaluator.state.metrics['overall_accuracy']
            logger.info("Validation Results - Epoch: {}".format(engine.state.epoch))
            logger.info("Precision: {} Average_Accuracy: {:.2f}".format(precision, avg_accuracy))
            logger.info("Overall_Accuracy: {:.3f}".format(overall_accuracy))
            """

    #5.engine运行
    trainer.run(train_loader, max_epochs=epochs)


