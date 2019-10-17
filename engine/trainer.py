# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import logging
try:
    # Capirca uses Google's abseil-py library, which uses a Google-specific
    # wrapper for logging. That wrapper will write a warning to sys.stderr if
    # the Google command-line flags library has not been initialized.
    #
    # https://github.com/abseil/abseil-py/blob/pypi-v0.7.1/absl/logging/__init__.py#L819-L825
    #
    # This is not right behavior for Python code that is invoked outside of a
    # Google-authored main program. Use knowledge of abseil-py to disable that
    # warning; ignore and continue if something goes wrong.
    import absl.logging

    # https://github.com/abseil/abseil-py/issues/99
    logging.root.removeHandler(absl.logging._absl_handler)
    # https://github.com/abseil/abseil-py/issues/102
    absl.logging._warn_preinit_stderr = False
except Exception:
    pass

import torch
import torch.nn as nn

import torchvision

from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer

from ignite.metrics import RunningAverage
from ignite.metrics import Accuracy
from ignite.metrics import Precision

from engine.evaluator import do_inference

from torch.utils.tensorboard import SummaryWriter

from solver import WarmupMultiStepLR
"""
try:
    from tensorboardX import SummaryWriter
except ImportError:
    raise RuntimeError(
        "No tensorboardX package is found. Please install with the command: \npip install tensorboardX")
"""


#CJY at 2019.9.24 既然这里面定义的与inference.py 中的一样能不能直接引用
#还没加

global ITER
ITER = 0

global accumulation_steps
accumulation_steps = 1

global epochs_traverse_optimizers
epochs_traverse_optimizers = 0

global epochs_per_optimizer
epochs_per_optimizer = 10

global op2loss
op2loss = {0:"D", 1:"G"}

global weight
weight = 1

def create_supervised_trainer(model, optimizers, metrics, loss_fn, device=None):
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
        for state in optimizers[0].state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

    epochs_traverse_optimizers = len(optimizers) * epochs_per_optimizer

    def _update(engine, batch):
        model.train()

        schedulers_epochs_index = (engine.state.epoch - 1) // epochs_traverse_optimizers
        index = (engine.state.epoch - 1) % epochs_traverse_optimizers  # 注意engine.state.epoch从1开始
        phase_index = index // epochs_per_optimizer
        # 记录在state中，方便传递给handler
        engine.state.optimizer_index = phase_index#(phase_index + 1) % len(optimizers)  #加了一个偏置，先训练G
        engine.state.schedulers_epochs_index = schedulers_epochs_index
        engine.state.epochs_traverse_optimizers = epochs_traverse_optimizers
        engine.state.losstype = op2loss[engine.state.optimizer_index]

        imgs, labels = batch   #这个格式应该跟collate_fn的处理方式对应
        imgs = imgs.to(device) if torch.cuda.device_count() >= 1 else imgs

        if labels.shape[0] > 1:
            if engine.state.losstype == "D":
                index1 = torch.arange(0, labels.shape[0]-1)
                labels1 = torch.index_select(labels, 0, index1, out=None)
                index2 = torch.arange(1, labels.shape[0])
                labels2 = torch.index_select(labels, 0, index2, out=None)
                similarity_labels = torch.eq(labels1, labels2).float()
                index3 = torch.arange(0, similarity_labels.shape[0], 2)
                similarity_labels = torch.index_select(similarity_labels, 0, index3, out=None).unsqueeze(1)
            else:
                similarity_labels = torch.ones(labels.shape[0]//2, 1).float()
            similarity_labels = similarity_labels.to(device) if torch.cuda.device_count() >= 1 else similarity_labels

        labels = labels.to(device) if torch.cuda.device_count() >= 1 else labels


        feats, scores, similarities ,sss = model(imgs)
        #loss = loss_fn(scores, feats, labels)/accumulation_steps
        #loss.backward()

        #利用不同的optimizer对模型中的各子模块进行分阶段优化。目前最简单的方式是周期循环启用optimizer
        if engine.state.iteration % accumulation_steps == 0:  #此处要注意

            losses = loss_fn[engine.state.losstype](feats, scores, labels, similarities, similarity_labels) #/ accumulation_steps
            #"""
            loss = 0
            for l in losses:
                loss += l
            #"""
            #loss = losses[1]
            loss.backward()
            optimizers[engine.state.optimizer_index].step()
            for op in optimizers:
                op.zero_grad()
        similarities = torch.gt(similarities, 0.5).int()
        """
        if engine.state.losstype == "D":
            similarity_loss = losses[1].item()
        else:
            similarity_loss = 0
        """
        # compute acc
        #acc = (scores.max(1)[1] == labels).float().mean()
        return {"scores":scores, "labels":labels, "classification_loss":losses[0].item(), "similarities":similarities, "similarity_labels":similarity_labels, "similarity_loss":losses[1].item(), "total_loss":loss.item()}

    engine = Engine(_update)


    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        classes_list,
        optimizers,
        schedulers,
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

    # TensorBoard setup
    writer_train = {}
    for i in range(len(optimizers)):
        writer_train[i] = SummaryWriter(cfg.SOLVER.OUTPUT_DIR + "/summary/train/" + str(i))
    writer_train["graph"] = SummaryWriter(cfg.SOLVER.OUTPUT_DIR + "/summary/train/graph")

    try:
        #print(model)
        images, labels = next(iter(train_loader))
        grid = torchvision.utils.make_grid(images)
        writer_train["graph"].add_image('images', grid, 0)
        writer_train["graph"].add_graph(model, images)
        writer_train["graph"].flush()
    except Exception as e:
        print("Failed to save model graph: {}".format(e))

    writer_val = SummaryWriter(cfg.SOLVER.OUTPUT_DIR + "/summary/val")


    # record graph
    #writer.add_graph(model)  #可以不加输入吗？加不加有什么不同

    metrics_train = {"avg_total_loss": RunningAverage(output_transform=lambda x: x["total_loss"]),
                     "avg_classification_loss": RunningAverage(output_transform=lambda x: x["classification_loss"]),
                     "avg_classification_precision": RunningAverage(Precision(output_transform=lambda x: (x["scores"], x["labels"]))),
                     "avg_classification_accuracy": RunningAverage(Accuracy(output_transform=lambda x: (x["scores"], x["labels"]))),  #由于训练集样本均衡后远离原始样本集，故只采用平均metric
                     "avg_similarity_loss": RunningAverage(output_transform=lambda x: x["similarity_loss"]),
                     "avg_similarity_accuracy": RunningAverage(Accuracy(output_transform=lambda x: (x["similarities"], x["similarity_labels"]))),}
    trainer = create_supervised_trainer(model, optimizers, metrics_train, loss_fn, device=device)

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
                                                                     'optimizer': optimizers[0]})
    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)


    #4.事件处理函数
    @trainer.on(Events.STARTED)
    def start_training(engine):
        engine.state.epoch = start_epoch
        """
        metrics = do_inference(cfg, model, val_loader, classes_list, loss_fn, plotFlag=False)

        step = 0#len(train_loader) * (engine.state.epoch - 1) + engine.state.iteration
        for preKey in metrics['precision'].keys():
            writer_val.add_scalar("Precision/" + str(preKey), metrics['precision'][preKey], step)

        for recKey in metrics['recall'].keys():
            writer_val.add_scalar("Recall/" + str(recKey), metrics['recall'][recKey], step)

        for aucKey in metrics['roc_auc'].keys():
            writer_val.add_scalar("ROC_AUC/" + str(aucKey), metrics['roc_auc'][aucKey], step)

        writer_val.add_scalar("OverallAccuracy", metrics["overall_accuracy"], step)

        # writer.add_scalar("Val/"+"confusion_matrix", metrics['confusion_matrix'], step)

        # 混淆矩阵 和 ROC曲线可以用图的方式来存储
        roc_numpy = metrics["roc_figure"]
        writer_val.add_image("ROC", roc_numpy, step, dataformats='HWC')

        confusion_matrix_numpy = metrics["confusion_matrix_numpy"]
        writer_val.add_image("ConfusionMatrix", confusion_matrix_numpy, step, dataformats='HWC')

        writer_val.flush()
        #"""

    @trainer.on(Events.EPOCH_COMPLETED) #_STARTED)   #注意，在pytorch1.2里面 scheduler.steo()应该放到 optimizer.step()之后
    def adjust_learning_rate(engine):
        """
        #if (engine.state.epoch - 1) % engine.state.epochs_traverse_optimizers == 0:
        if engine.state.epoch == 2:
            op_i_scheduler1 = WarmupMultiStepLR(optimizers[0], cfg.SOLVER.SCHEDULER.STEPS, cfg.SOLVER.SCHEDULER.GAMMA,
                                               cfg.SOLVER.SCHEDULER.WARMUP_FACTOR,
                                               cfg.SOLVER.SCHEDULER.WARMUP_ITERS, cfg.SOLVER.SCHEDULER.WARMUP_METHOD)
            op_i_scheduler2 = WarmupMultiStepLR(optimizers[1], cfg.SOLVER.SCHEDULER.STEPS, cfg.SOLVER.SCHEDULER.GAMMA,
                                                cfg.SOLVER.SCHEDULER.WARMUP_FACTOR,
                                                cfg.SOLVER.SCHEDULER.WARMUP_ITERS, cfg.SOLVER.SCHEDULER.WARMUP_METHOD)
            engine.state.schedulers = [op_i_scheduler1, op_i_scheduler2]
            print("copy")
        """
        schedulers[engine.state.schedulers_epochs_index][engine.state.optimizer_index].step()

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        global ITER
        ITER += 1

        if ITER % (log_period*accumulation_steps) == 0:
            avg_precision = engine.state.metrics['avg_classification_precision'].numpy().tolist()
            for index, ap in enumerate(avg_precision):
                avg_precision[index] = float("{:.2f}".format(ap))

            step = engine.state.iteration

            logger.info("Epoch[{}] Iteration[{}/{}] Training {} - Avg_C_Loss: {:.3f}, Avg C_Pre: {}, Avg_C Acc: {:.3f}, Avg_S_loss: {:.3f}, Avg_S_Acc: {:.3f}, Base Lr: {:.2e}, step: {}"
                        .format(engine.state.epoch, ITER, len(train_loader),
                                engine.state.losstype,
                                engine.state.metrics['avg_classification_loss'], avg_precision, engine.state.metrics['avg_classification_accuracy'],
                                engine.state.metrics['avg_similarity_loss'], engine.state.metrics['avg_similarity_accuracy'],
                                schedulers[engine.state.schedulers_epochs_index][engine.state.optimizer_index].get_lr()[0], step))

            scalar_list = ["avg_classification_loss", "avg_classification_accuracy", "avg_similarity_loss", "avg_similarity_accuracy", "avg_total_loss"]
            #for scalar in scalar_list:
            #    writer_train.add_scalar("Train/"+scalar, engine.state.metrics[scalar], step)
            #writer_train.add_scalar("Train/" + "LearningRate1", schedulers[0].get_lr()[0], step)

            for scalar in scalar_list:
                scalarDict = {}
                for i in range(len(optimizers)):
                    if i != engine.state.optimizer_index:
                        scalarDict["optimizer" + str(i)] = 0
                    else:
                        scalarDict["optimizer" + str(i)] = engine.state.metrics[scalar]
                    writer_train[i].add_scalar("Train/" + scalar, scalarDict["optimizer" + str(i)], step)
                    writer_train[i].flush()
                #writer_train.add_scalars("Train/"+scalar, scalarDict, step)

            LearningRateDict = {}
            for i in range(len(optimizers)):
                if i != engine.state.optimizer_index:
                    LearningRateDict["optimizer" + str(i)] = 0
                else:
                    LearningRateDict["optimizer" + str(i)] = schedulers[engine.state.schedulers_epochs_index][engine.state.optimizer_index].get_lr()[0]
                writer_train[i].add_scalar("Train/" + "LearningRate", LearningRateDict["optimizer" + str(i)], step)
                writer_train[i].flush()

            #加入weight
            choose_list = ["base.conv1.weight", "base.bn1.weight",
                          "base.layer1.0.conv1.weight", "base.layer1.2.conv3.weight",
                          "base.layer2.0.conv1.weight", "base.layer2.3.conv3.weight",
                          "base.layer3.0.conv1.weight", "base.layer3.5.conv3.weight",
                          "base.layer4.0.conv1.weight", "base.layer4.2.conv1.weight",
                          "bottleneck.weight", "classifier.weight"]
            params_dict = {}
            for name, parameters in model.named_parameters():
                #print(name, ':', parameters.size())
                params_dict[name] = parameters.detach().cpu().numpy()
            #print(len(params_dict))

            for cp in params_dict.keys():
                writer_train["graph"].add_histogram("Train/" + cp, params_dict[cp], step)
                writer_train["graph"].flush()
            #writer_train.add_scalars("Train/" + "LearningRate", LearningRateDict, step)
            #writer_train.flush()

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
            metrics = do_inference(cfg, model, val_loader, classes_list, loss_fn)

            step = engine.state.iteration
            for preKey in metrics['precision'].keys():
                writer_val.add_scalar("Precision/" + str(preKey), metrics['precision'][preKey], step)

            for recKey in metrics['recall'].keys():
                writer_val.add_scalar("Recall/" + str(recKey), metrics['recall'][recKey], step)

            for aucKey in metrics['roc_auc'].keys():
                writer_val.add_scalar("ROC_AUC/" + str(aucKey), metrics['roc_auc'][aucKey], step)

            writer_val.add_scalar("OverallAccuracy", metrics["overall_accuracy"], step)

            #writer.add_scalar("Val/"+"confusion_matrix", metrics['confusion_matrix'], step)

            #混淆矩阵 和 ROC曲线可以用图的方式来存储
            roc_numpy = metrics["roc_figure"]
            writer_val.add_image("ROC", roc_numpy, step, dataformats='HWC')

            confusion_matrix_numpy = metrics["confusion_matrix_numpy"]
            writer_val.add_image("ConfusionMatrix", confusion_matrix_numpy, step, dataformats='HWC')

            writer_val.flush()


    #5.engine运行
    trainer.run(train_loader, max_epochs=epochs)
    for key in writer_train.keys():
        writer_train[key].close()
    writer_val.close()
