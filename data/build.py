# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from torch.utils.data import DataLoader

#CJY
from torchvision.datasets import ImageFolder

from .collate_batch import train_collate_fn, val_collate_fn
from .collate_batch import train_collate_fn_classifaction, val_collate_fn_classifaction, test_collate_fn_classifaction
from .datasets import init_dataset, ImageDataset, ImageDatasetForClassification
from .samplers import RandomIdentitySampler, RandomSampler, RandomSampler_NonTrain, RandomIdentitySampler_alignedreid  # New add by gu
from .transforms import build_transforms


def make_data_loader(cfg):
    train_transforms = build_transforms(cfg, is_train=True)
    val_transforms = build_transforms(cfg, is_train=False)
    num_workers = cfg.DATALOADER.NUM_WORKERS
    if len(cfg.DATASETS.NAMES) == 1:
        dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR)
    else:
        # TODO: add multi dataset to train
        dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR)

    num_classes = dataset.num_train_pids
    train_set = ImageDataset(dataset.train, train_transforms)
    if cfg.DATA.DATALOADER.SAMPLER == 'softmax':
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn
        )
    else:
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
            sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
            # sampler=RandomIdentitySampler_alignedreid(dataset.train, cfg.DATALOADER.NUM_INSTANCE),      # new add by gu
            num_workers=num_workers, collate_fn=train_collate_fn
        )

    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    return train_loader, val_loader, len(dataset.query), num_classes

def make_data_loader_classification(cfg):
    train_transforms = build_transforms(cfg, is_train=True)
    val_transforms = build_transforms(cfg, is_train=False)
    test_transforms = build_transforms(cfg, is_train=False)

    num_workers = cfg.DATA.DATALOADER.NUM_WORKERS
    if len(cfg.DATA.DATASETS.NAMES) == 1:
        dataset = init_dataset(cfg.DATA.DATALOADER.NAMES, root=cfg.DATA.DATALOADER.ROOT_DIR)
    else:
        # TODO: add multi dataset to train
        dataset = init_dataset(cfg.DATA.DATASETS.NAMES, root=cfg.DATA.DATASETS.ROOT_DIR)

    num_classes = dataset.num_categories

    #train set
    #是否要进行label-smoothing
    #train_set = ImageDataset(dataset.train, train_transforms)
    train_set = ImageDatasetForClassification(dataset.train, train_transforms)
    if cfg.DATA.DATALOADER.SAMPLER == 'softmax':
        train_loader = DataLoader(
            train_set, batch_size=cfg.TRAIN.DATALOADER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn_classifaction
        )
    else:
        train_loader = DataLoader(
            train_set, batch_size=cfg.TRAIN.DATALOADER.IMS_PER_BATCH,
            sampler=RandomSampler(dataset.train, cfg.TRAIN.DATALOADER.CATEGORIES_PER_BATCH, cfg.TRAIN.DATALOADER.INSTANCES_PER_CATEGORY_IN_BATCH, dataset.num_categories, is_train=True),
            # sampler=RandomIdentitySampler_alignedreid(dataset.train, cfg.DATALOADER.NUM_INSTANCE),      # new add by gu
            num_workers=num_workers, collate_fn=train_collate_fn_classifaction
        )

    #val set
    #val_set = ImageDataset(dataset.val, val_transforms)
    val_set = ImageDatasetForClassification(dataset.val, val_transforms)
    val_loader = DataLoader(
        val_set, batch_size=cfg.VAL.DATALOADER.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        #CJY at 2019.9.26 为了能够平衡样本
        sampler=RandomSampler(dataset.val, cfg.VAL.DATALOADER.CATEGORIES_PER_BATCH, cfg.VAL.DATALOADER.INSTANCES_PER_CATEGORY_IN_BATCH, dataset.num_categories, is_train=False),
        collate_fn=val_collate_fn_classifaction
    )

    #test_set
    #test_set = ImageDataset(dataset.test, test_transforms)
    test_set = ImageDatasetForClassification(dataset.test, test_transforms)
    test_loader = DataLoader(
        test_set, batch_size=cfg.TEST.DATALOADER.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        # CJY at 2019.9.26 为了能够平衡样本
        sampler=RandomSampler(dataset.test, cfg.TEST.DATALOADER.CATEGORIES_PER_BATCH, cfg.TEST.DATALOADER.INSTANCES_PER_CATEGORY_IN_BATCH, dataset.num_categories, is_train=False),
        collate_fn=test_collate_fn_classifaction
    )
    #notes:
    #1.collate_fn是自定义函数，对提取的batch做处理，例如分开image和label

    return train_loader, val_loader, test_loader, num_classes
