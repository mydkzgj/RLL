# encoding: utf-8
"""
@author:  liaoxingyu
@contact: liaoxingyu2@jd.com
"""

import copy
import random
import torch
from collections import defaultdict

import numpy as np
from torch.utils.data.sampler import Sampler


class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)   #有放回抽样到num_instance的数量
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:    #将idxs按num_instances数量进行分组，结尾不够num_instances的在这个epoch就不要了
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)   #随机挑选人
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length

#CJY at 2019.9.26
class RandomSampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, label).
    - num_categories_per_batch (int): number of categories per batch.
    - num_instances_per_category (int): number of instances per category in a batch.
    - batch_size (int): number of examples in a batch.
    -
    """

    def __init__(self, data_source, num_categories_per_batch, num_instances_per_category, max_num_categories, is_train=True):
        self.data_source = data_source
        self.num_categories_per_batch = num_categories_per_batch
        self.num_instances_per_category = num_instances_per_category
        self.batch = num_categories_per_batch * num_instances_per_category
        self.is_train = is_train

        #num_categories_per_batch不能小于总的类别数
        if self.num_categories_per_batch > max_num_categories or self.num_categories_per_batch < 2:
            raise Exception("Invalid Num_categories_per_batch!", self.num_categories_per_batch)

        #将data_source中的samples依照类别将同类的sample以列表的形式存入字典中
        self.index_dic = defaultdict(list)  #这种字典与普通字典的却别？
        for index, (_, label) in enumerate(self.data_source):
            self.index_dic[label].append(index)
        self.categories = list(self.index_dic.keys())

        #记录每类的sample数量，并找出最大sample数量的类别（用于后续平衡其他类别的标准）
        self.targetNum_instances_per_category = {}
        max_num_samples = 0
        min_num_samples = 10000000
        for category in self.index_dic.keys():
            self.targetNum_instances_per_category[category] = len(self.index_dic[category])
            if max_num_samples < self.targetNum_instances_per_category[category]:
                max_num_samples =  self.targetNum_instances_per_category[category]
            if min_num_samples > self.targetNum_instances_per_category[category]:
                min_num_samples =  self.targetNum_instances_per_category[category]
        if max_num_samples % self.num_instances_per_category == 0:  #保证每类的samples含有整倍数的instances
            self.max_num_samples = max_num_samples
        else:
            self.max_num_samples = (max_num_samples//self.num_instances_per_category + 1) * self.num_instances_per_category
        self.max_num_samples = 100 * self.num_instances_per_category   #减少训练集样本数量，加快训练速度
        self.min_num_samples = min_num_samples

        #设置每类的样本需要构建数量
        if self.is_train == True:
            for category in self.index_dic.keys():
                self.targetNum_instances_per_category[category] = self.max_num_samples
        else:
            #为了加快验证时间所做，后期移除
            for category in self.index_dic.keys():
                self.targetNum_instances_per_category[category] = self.min_num_samples

        #epoch内样本数量
        self.length = 0
        for category in self.index_dic.keys():
            self.length = self.length + self.targetNum_instances_per_category[category]


    def __iter__(self):  #核心函数，返回一个迭代器
        # 将instances按每self.num_instances_per_category为一组存储到类别索引的字典里
        batch_idxs_dict = defaultdict(list)

        for category in self.categories:
            # 1.首先通过循环串联每类的样本（乱序）来增加该类的待选样本长度
            idxs = []
            while len(idxs) < self.targetNum_instances_per_category[category]:
                copy_idxs = copy.deepcopy(self.index_dic[category])
                random.shuffle(copy_idxs)
                idxs = idxs + copy_idxs
            idxs = idxs[0 : self.targetNum_instances_per_category[category]]

            #2.将每类样本按每self.num_instances_per_category一组分割（即一个batch内的instance）
            batch_idxs = []
            for i, idx in enumerate(idxs):
                batch_idxs.append(idx)
                if i == self.targetNum_instances_per_category[category]-1:  #不舍弃后续不足一个batch内instance—per-category的样本（非训练情形）
                    batch_idxs_dict[category].append(batch_idxs)
                    batch_idxs = []
                    break
                if len(batch_idxs) == self.num_instances_per_category:
                    batch_idxs_dict[category].append(batch_idxs)
                    batch_idxs = []

        #随机抽取组成batch，即设定迭代计划
        copy_categories = copy.deepcopy(self.categories)
        final_idxs = []   #迭代器核心列表
        if self.is_train == True:
            num_categories_th = self.num_categories_per_batch - 1
            while len(copy_categories) > num_categories_th:   #若其小于每个batch需要抽取的class则停止
                if self.is_train == True:
                    selected_categories = random.sample(copy_categories, self.num_categories_per_batch)   #随机挑选类别

                batch_idxs = []
                for category in selected_categories:
                    batch_idxs += batch_idxs_dict[category].pop(0)
                    if len(batch_idxs_dict[category]) == 0:
                        copy_categories.remove(category)
                batch_idxs1 = batch_idxs[0:len(batch_idxs)//4]
                batch_idxs2 = batch_idxs[len(batch_idxs)//4:len(batch_idxs)]
                random.shuffle(batch_idxs2)
                batch_idxs = batch_idxs1 + batch_idxs2
                final_idxs.extend(batch_idxs)
        else:
            num_categories_th = 0
            while len(copy_categories) > num_categories_th:  # 若其小于每个batch需要抽取的class则停止
                if len(copy_categories) >= self.num_categories_per_batch:
                    selected_categories = np.random.choice(copy_categories, self.num_categories_per_batch, replace=False)  # 随机挑选类别
                else:
                    selected_categories = np.random.choice(copy_categories, self.num_categories_per_batch, replace=True)  # 随机挑选类别

                for category in selected_categories:
                    if len(batch_idxs_dict[category]) <= 0:
                        continue
                    batch_idxs = batch_idxs_dict[category].pop(0)
                    final_idxs.extend(batch_idxs)
                    if len(batch_idxs_dict[category]) == 0:
                        copy_categories.remove(category)
        self.epoch_num_samples = len(final_idxs)

        return iter(final_idxs)

    def __len__(self):
        return self.length


# New add by gu
class RandomIdentitySampler_alignedreid(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/data/sampler.py.

    Args:
        data_source (Dataset): dataset to sample from.
        num_instances (int): number of instances per identity.
    """
    def __init__(self, data_source, num_instances):
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid, _) in enumerate(data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_identities = len(self.pids)

    def __iter__(self):
        indices = torch.randperm(self.num_identities)
        ret = []
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]
            replace = False if len(t) >= self.num_instances else True
            t = np.random.choice(t, size=self.num_instances, replace=replace)
            ret.extend(t)
        return iter(ret)

    def __len__(self):
        return self.num_identities * self.num_instances
