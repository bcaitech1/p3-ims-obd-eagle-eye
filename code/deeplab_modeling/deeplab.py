import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.insert(1, '/opt/ml/p3-ims-obd-eagle-eye/code/deeplab_modeling')
import os
# import deeplab_modeling
from sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from aspp import build_aspp
from decoder import build_decoder
from backbone import build_backbone

class DeepLab(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False):
        super(DeepLab, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)

        self.freeze_bn = freeze_bn

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

if __name__ == "__main__":
    model = DeepLab(backbone='resnet',num_classes=12)
    # print(model)
    checkpoint = torch.load(os.path.join('/opt/ml/p3-ims-obd-eagle-eye/pretrained','deeplab-resnet.pth.tar'))
    # print(checkpoint['state_dict']['decoder.last_conv.8.weight'].shape)
    # for n,parms in checkpoint['state_dict'].items():
    #     print(n,parms.shape)
    #     # break
    print(torch.max(checkpoint['state_dict']['decoder.last_conv.8.weight']))
    tensor = torch.empty(12,256,1,1)
    print(torch.max(torch.nn.init.xavier_normal_(tensor, gain=1.0)))
    # new_weight=torch.FloatTensor(12,256,1,1).uniform_(torch.min(checkpoint['state_dict']['decoder.last_conv.8.weight']), torch.max(checkpoint['state_dict']['decoder.last_conv.8.weight']))
    # print(torch.mean(new_weight)*1000)
    # checkpoint['state_dict']['decoder.last_conv.8.weight']=new_weight
    # checkpoint['state_dict']['decoder.last_conv.8.bias']=torch.FloatTensor(12).uniform_(torch.min(checkpoint['state_dict']['decoder.last_conv.8.bias']), torch.max(checkpoint['state_dict']['decoder.last_conv.8.bias']))
    # # checkpoint['state_dict']['decoder.last_conv.8.bias']=torch.rand(12)
    # # print(checkpoint['state_dict']['decoder.last_conv.8.weight'].shape)
    # model.load_state_dict(checkpoint['state_dict'])
    # model.eval()
    # input = torch.rand(1, 3, 513, 513)
    # output = model(input)
    # print(output.size())


