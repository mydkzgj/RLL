# encoding: utf-8
"""
@author:  cjy
@contact: sychenjiayang@163.com
"""

import torch
from collections import defaultdict

# 创建多个optimizer，用来交替训练模型的各个子部分
def make_optimizers(cfg, model):
    params_dict2 = {}
    for name, parameters in model.named_parameters():
        # print(name, ':', parameters.size())
        params_dict2[name] = parameters.detach().cpu().numpy()


    groupKeys = ["base.conv1", "base.bn1", "base.layer1", "base.layer2", "base.layer3", "base.layer4", "bottleneck", "classifier", "others"]
    params_dict = defaultdict(list)
    parameters = model.named_parameters()
    out_include = []
    for key, value in parameters:
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.SCHEDULER.BASE_LR
        weight_decay = cfg.SOLVER.OPTIMIZER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.SCHEDULER.BASE_LR * cfg.SOLVER.SCHEDULER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.OPTIMIZER.WEIGHT_DECAY_BIAS
        for gkey in groupKeys:
            if gkey in key or gkey == "others":
                if gkey in ["base.conv1", "base.bn1", "base.layer1", "base.layer2", "base.layer3"]:
                    params_dict[gkey].append({"params": [value], "lr": lr, "weight_decay": weight_decay})
                else:
                    params_dict[gkey].append({"params": [value], "lr": lr, "weight_decay": weight_decay})
                break



    gkeys_divided_list = [[0,1,2,3,4,5,6,7,8],[0,1,2,3]] #[0,1,2,3,4,5,6,7],  3,4,5,6,7,8
    params_divided_list = []
    for sub in gkeys_divided_list:
        p = []
        for i in sub:
            p += params_dict[groupKeys[i]]
        params_divided_list.append(p)

    optimizers = []
    if cfg.SOLVER.OPTIMIZER.NAME == 'SGD':
        for pd in params_divided_list:
            optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER.NAME)(pd , momentum=cfg.SOLVER.OPTIMIZER.MOMENTUM)
            optimizers.append(optimizer)
    else:
        for pd in params_divided_list:
            optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER.NAME)(pd)
            optimizers.append(optimizer)

    return optimizers