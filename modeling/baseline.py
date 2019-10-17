# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from torch import nn
import torch.nn.functional as F

from .backbones.resnet import ResNet, BasicBlock, Bottleneck
from .backbones.senet import SENet, SEResNetBottleneck, SEBottleneck, SEResNeXtBottleneck
from .backbones.mobilenetv3 import MobileNetV3_Large
from .backbones.resnet_ibn_a import resnet50_ibn_a
from .backbones.resnet_ibn_a_old import resnet50_ibn_a_old

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

class Baseline(nn.Module):
    in_planes = 2048

    def __init__(self,  base_name, num_classes, stn_flag, last_stride):
        super(Baseline, self).__init__()

        self.base_name = base_name

        if base_name == 'resnet18':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride, 
                               block=BasicBlock, 
                               layers=[2, 2, 2, 2])
        elif base_name == 'resnet34':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride,
                               block=BasicBlock,
                               layers=[3, 4, 6, 3])
        elif base_name == 'resnet50':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
        elif base_name == 'resnet101':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck, 
                               layers=[3, 4, 23, 3])
        elif base_name == 'resnet152':
            self.base = ResNet(last_stride=last_stride, 
                               block=Bottleneck,
                               layers=[3, 8, 36, 3])
        elif base_name == 'se_resnet50':
            self.base = SENet(block=SEResNetBottleneck, 
                              layers=[3, 4, 6, 3], 
                              groups=1, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride) 
        elif base_name == 'se_resnet101':
            self.base = SENet(block=SEResNetBottleneck, 
                              layers=[3, 4, 23, 3], 
                              groups=1, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride)
        elif base_name == 'se_resnet152':
            self.base = SENet(block=SEResNetBottleneck, 
                              layers=[3, 8, 36, 3],
                              groups=1, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride)  
        elif base_name == 'se_resnext50':
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 6, 3], 
                              groups=32, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride) 
        elif base_name == 'se_resnext101':
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 23, 3], 
                              groups=32, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride)
        elif base_name == 'senet154':
            self.base = SENet(block=SEBottleneck, 
                              layers=[3, 8, 36, 3],
                              groups=64, 
                              reduction=16,
                              dropout_p=0.2, 
                              last_stride=last_stride)
                              
        elif base_name == 'mobilenetv3':
            self.in_planes = 960
            self.base = MobileNetV3_Large()

        elif base_name == 'resnet50_ibn_a':
            self.base = resnet50_ibn_a(last_stride = last_stride)

        elif base_name == 'resnet50_ibn_a_old':
            self.base = resnet50_ibn_a_old(last_stride = last_stride)

        # 2.以下是stn的网络结构
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 60 * 28, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        # 3.以下是classifier的网络结构
        self.gap = nn.AdaptiveAvgPool2d(1)
        # self.gap = nn.AdaptiveMaxPool2d(1)
        self.num_classes = num_classes
        self.stn_flag = stn_flag

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

        #
        self.scroe_by_class = torch.nn.Sigmoid()

        # 4.自建，图像为同类或异类的判别器
        """
        self.similarity_classifier = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(2,1)),
            nn.ReLU(True),
            nn.Conv2d(8, 1, kernel_size=1),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(self.in_planes, 1, bias=False),   #有两排
            #nn.ReLU(True),
            #nn.Linear(32, 1, bias=False),
            nn.Sigmoid()
        )
        self.similarity_classifier[5].apply(weights_init_classifier)
        """
        self.s_conv1 = nn.Conv2d(1, 8, kernel_size=(2,1))
        self.s_relu1 = nn.ReLU(True)
        self.s_conv2 = nn.Conv2d(8, 1, kernel_size=1)
        self.s_relu2 = nn.ReLU(True)
        self.s_flat = nn.Flatten()
        self.s_classifier = nn.Linear(self.in_planes, 1)#, bias=False)
        self.s_s = nn.Sigmoid()

        
    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 60 * 28)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x
        
    def forward(self, x):
        if self.stn_flag == 'yes':
            x = self.stn(x)    #### stn

        global_feat = self.gap(self.base(x))  # (b, 2048, 1, 1)

        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

        feat = self.bottleneck(global_feat)  # normalize for angular softmax

        # CJY at 2019.10.8 去掉if self.training:,因为我们最终需要的模型就是classifier
        cls_score = self.classifier(feat)

        #cls_score_sig = self.scroe_by_class(cls_score)



        #CJY at 2019.10.15
        pairwise_global_feat = global_feat.view((global_feat.shape[0]//2, 1, -1, global_feat.shape[1]))
        #pairwise_global_feat = global_feat.view(global_feat.shape[0]//2, -1)
        """
        index1 = torch.arange(0, global_feat.shape[0], 2).cuda()
        global_feat1 = torch.index_select(global_feat, 0, index1, out=None)
        index2 = torch.arange(1, global_feat.shape[0], 2).cuda()
        global_feat2 = torch.index_select(global_feat, 0, index2, out=None)

        #if global_feat1.shape[0] != global_feat2.shape[0]:
        #    raise Exception("batch size isn't even")
        pairwise_global_feat = torch.abs(torch.add(global_feat1, -1, global_feat2))
        #"""

        similarity = self.similarity_classifier(pairwise_global_feat)
        #"""
        sc1 = self.s_conv1(pairwise_global_feat)
        sr1 = self.s_relu1(sc1)
        sc2 = self.s_conv2(sr1)
        sr2 = self.s_relu2(sc2)
        sf = self.s_flat(sr2)
        scl = self.s_classifier(sf)
        similarity = self.s_s(scl)
        

        sss = {"global_feat": global_feat, "pairwise_global_feat":pairwise_global_feat, "sc1":sc1, "sr1":sc1, "sc2":sc2, "sr2":sc2, "sf":sf, "scl":scl}
        #"""


        return feat, cls_score, similarity, sss ### feat for ranked loss   #实际上我们是否可以引出很多feat


    def load_param(self, loadChoice, model_path):
        param_dict = torch.load(model_path)

        if loadChoice == "Base":
            for i in param_dict:
                if i not in self.base.state_dict():  #注意，我们是需要classifier层参数的
                    continue
                self.base.state_dict()[i].copy_(param_dict[i])

        elif loadChoice == "Overall":   #CJY at 2019.10.4
            for i in param_dict:
                if i not in self.state_dict(): #or 'classifier' in i:  #注意，我们是需要classifier层参数的
                    continue
                self.state_dict()[i].copy_(param_dict[i])

        elif loadChoice == "Classifier":   #CJY at 2019.10.4
            for i in param_dict:
                if i not in self.classifier.state_dict(): #or 'classifier' in i:  #注意，我们是需要classifier层参数的
                    continue
                self.classifier.state_dict()[i].copy_(param_dict[i])
