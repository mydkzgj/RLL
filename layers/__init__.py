# encoding: utf-8

import torch.nn.functional as F

from .reanked_loss import RankedLoss, CrossEntropyLabelSmooth
from .reanked_clu_loss import CRankedLoss

def make_loss(cfg, num_classes):
    sampler = cfg.DATA.DATALOADER.SAMPLER
    if cfg.LOSS.TYPE == 'ranked_loss':
        ranked_loss = RankedLoss(cfg.LOSS.MARGIN_RANK, cfg.LOSS.ALPHA, cfg.LOSS.TVAL) # ranked_loss
        
    elif cfg.LOSS.TYPE == 'cranked_loss':
        cranked_loss = CRankedLoss(cfg.LOSS.MARGIN_RANK, cfg.LOSS.ALPHA, cfg.LOSS.TVAL) # cranked_loss
        
    else:
        print('expected METRIC_LOSS_TYPE should be triplet, cluster, triplet_cluster'
              'but got {}'.format(cfg.LOSS.TYPE))

    if cfg.TRAIN.TRICK.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)     # new add by luo
        print("label smooth on, numclasses:", num_classes)

    if sampler == 'softmax':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)
    elif cfg.DATA.DATALOADER.SAMPLER == 'ranked_loss':
        def loss_func(score, feat, target):
            #return ranked_loss(feat, target)[0]
            return ranked_loss(feat, target)
    elif cfg.DATA.DATALOADER.SAMPLER == 'cranked_loss':
        def loss_func(score, feat, target):
            #return cranked_loss(feat, target)[0]
            return cranked_loss(feat, target)
    elif cfg.DATA.DATALOADER.SAMPLER == 'softmax_rank':
        def loss_func(score, feat, target):
            if cfg.LOSS.TYPE == 'ranked_loss':
                if cfg.TRAIN.TRICK.IF_LABELSMOOTH == 'on':
                    #return  xent(score, target) + cfg.SOLVER.WEIGHT*ranked_loss(feat, target)[0] # new add by zzg, open label smooth
                    return xent(score, target) + cfg.LOSS.WEIGHT * ranked_loss(feat, target)  #CJY at 2019.9.23, 这个改动与pytorch版本有关
                else:
                    #return F.cross_entropy(score, target) + ranked_loss(feat, target)[0]    # new add by zzg, no label smooth
                    return F.cross_entropy(score, target) + ranked_loss(feat, target)  #CJY at 2019.9.23, 这个改动与pytorch版本有关

            elif cfg.LOSS.TYPE == 'cranked_loss':
                if cfg.TRAIN.TRICK.IF_LABELSMOOTH == 'on':
                    #return  xent(score, target) +cfg.SOLVER.WEIGHT*cranked_loss(feat, target)[0] # new add by zzg, open label smooth
                    return xent(score, target) + cfg.LOSS.WEIGHT * cranked_loss(feat, target) #CJY at 2019.9.23, 这个改动与pytorch版本有关
                else:
                    #return F.cross_entropy(score, target) + cranked_loss(feat, target)[0]    # new add by zzg, no label smooth
                    return F.cross_entropy(score, target) + cranked_loss(feat, target)  #CJY at 2019.9.23, 这个改动与pytorch版本有关
            else:
                print('expected METRIC_LOSS_TYPE should be triplet, cluster, triplet_cluster，'
                      'but got {}'.format(cfg.LOSSL.TYPE))
    else:
        print('expected sampler should be softmax, ranked_loss or cranked_loss, '
              'but got {}'.format(cfg.DATA.DATALOADER.SAMPLER))
    return loss_func


