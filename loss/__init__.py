# encoding: utf-8

import torch.nn.functional as F

from .reanked_loss import RankedLoss, CrossEntropyLabelSmooth
from .reanked_clu_loss import CRankedLoss
from .common_loss import CommonLoss
from .similarity_loss import SimilarityLoss


def make_D_loss(cfg, num_classes):
    sampler = cfg.DATA.DATALOADER.SAMPLER

    lossKeys = cfg.LOSS.TYPE.split(" ")
    for lossName in lossKeys:
        if lossName == "similarity_loss":
            similarity_loss = SimilarityLoss()
        elif lossName == "ranked_loss":
            ranked_loss = RankedLoss(cfg.LOSS.MARGIN_RANK, cfg.LOSS.ALPHA, cfg.LOSS.TVAL)  # ranked_loss
        elif lossName == "cranked_loss":
            cranked_loss = CRankedLoss(cfg.LOSS.MARGIN_RANK, cfg.LOSS.ALPHA, cfg.LOSS.TVAL)  # cranked_loss
        else:
            raise Exception('expected METRIC_LOSS_TYPE should be similarity_loss, ranked_loss, cranked_loss'
              'but got {}'.format(cfg.LOSS.TYPE))

    if cfg.TRAIN.TRICK.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)  # new add by luo
        print("label smooth on, numclasses:", num_classes)

    def D_loss_func(feat, score, label, similarity, similarity_label):
        # return ranked_loss(feat, target)[0]
        """
        total_loss = 0
        total_loss = total_loss + F.cross_entropy(score, label)
        total_loss = total_loss + similarity_loss(similarity, similarity_label)
        """
        return [F.cross_entropy(score, label), similarity_loss(similarity, similarity_label)]

    """
    if sampler == 'softmax':
        def G_loss_func(score, feat, target):
            return F.cross_entropy(score, target)
    elif cfg.DATA.DATALOADER.SAMPLER == 'ranked_loss':
        def G_loss_func(score, feat, target):
            # return ranked_loss(feat, target)[0]
            return ranked_loss(feat, target, normalize_feature=False)
    elif cfg.DATA.DATALOADER.SAMPLER == 'cranked_loss':
        def G_loss_func(score, feat, target):
            # return cranked_loss(feat, target)[0]
            return cranked_loss(feat, target)
    elif cfg.DATA.DATALOADER.SAMPLER == 'softmax_rank':
        def G_loss_func(score, feat, target):
            if cfg.LOSS.TYPE == 'ranked_loss':
                if cfg.TRAIN.TRICK.IF_LABELSMOOTH == 'on':
                    # return  xent(score, target) + cfg.SOLVER.WEIGHT*ranked_loss(feat, target)[0] # new add by zzg, open label smooth
                    return xent(score, target) + cfg.LOSS.WEIGHT * ranked_loss(feat,
                                                                               target, normalize_feature=False)  # CJY at 2019.9.23, 这个改动与pytorch版本有关
                else:
                    # return F.cross_entropy(score, target) + ranked_loss(feat, target)[0]    # new add by zzg, no label smooth
                    return F.cross_entropy(score, target) + ranked_loss(feat,
                                                                        target, normalize_feature=False)  # CJY at 2019.9.23, 这个改动与pytorch版本有关

            elif cfg.LOSS.TYPE == 'cranked_loss':
                if cfg.TRAIN.TRICK.IF_LABELSMOOTH == 'on':
                    # return  xent(score, target) +cfg.SOLVER.WEIGHT*cranked_loss(feat, target)[0] # new add by zzg, open label smooth
                    return xent(score, target) + cfg.LOSS.WEIGHT * cranked_loss(feat,
                                                                                target)  # CJY at 2019.9.23, 这个改动与pytorch版本有关
                else:
                    # return F.cross_entropy(score, target) + cranked_loss(feat, target)[0]    # new add by zzg, no label smooth
                    return F.cross_entropy(score, target) + cranked_loss(feat,
                                                                         target)  # CJY at 2019.9.23, 这个改动与pytorch版本有关
            else:
                print('expected METRIC_LOSS_TYPE should be triplet, cluster, triplet_cluster，'
                      'but got {}'.format(cfg.LOSSL.TYPE))
    else:
        print('expected sampler should be softmax, ranked_loss or cranked_loss, '
              'but got {}'.format(cfg.DATA.DATALOADER.SAMPLER))
    """

    return D_loss_func

def make_G_loss(cfg, num_classes):   #注意是找相同点common
    """
    common_loss = CommonLoss(num_classes=num_classes, margin=cfg.LOSS.MARGIN_RANK)  # common_loss
    def D_loss_func(score, feat, target):
        return common_loss(score, target)
    """
    """
    if cfg.LOSS.TYPE == 'ranked_loss':
        common_loss = CommonLoss(cfg.LOSS.MARGIN_RANK, cfg.LOSS.ALPHA, cfg.LOSS.TVAL)  # ranked_loss
    def D_loss_func(score, feat, target):
        return common_loss(feat, target, normalize_feature=False)
    """
    common_loss = CommonLoss(cfg.LOSS.MARGIN_RANK, cfg.LOSS.ALPHA, cfg.LOSS.TVAL)  # ranked_loss
    similarity_loss = SimilarityLoss()
    def G_loss_func(feat, score, label, similarity, similarity_label):
        return [F.cross_entropy(score, label), similarity_loss(similarity, similarity_label)]

    return G_loss_func

