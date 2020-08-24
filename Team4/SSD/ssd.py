
# coding: utf-8

# In[75]:


import math
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable


# In[76]:


__all__ = ['ResNetBase', 'resnet_base']

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


# In[77]:


class ConvBnReluLayer(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, stride, bias=False):
        super(ConvBnReluLayer, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, padding=padding, stride=stride, bias=bias)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


# In[78]:


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
  # 3x3 conv with padding
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, dilation=dilation, padding=1, bias=False)


# In[79]:


class BasicBlock(nn.Module):
    expansion = 1
  
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        residual = x
    
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual  
        out = self.relu(out)

        return out


# In[80]:


class Bottleneck(nn.Module):
    expansion = 4
  
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    
    
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# In[81]:


class ResNetBase(nn.Module):
    def __init__(self, block, layers, width=1, num_classes=2):
        self.inplanes = 64
        widths = [int(round(ch * width)) for ch in [64, 128, 256, 512]]

        super(ResNetBase, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, widths[0], layers[0])
        self.layer2 = self._make_layer(block, widths[1], layers[1], stride=2)
        #self.layer3 = self._make_layer(block, widths[2], layers[2], stride=2)
        # change stride = 2, dilation = 1 in ResNet to stride = 1, dilation = 2 for the final _make_layer
        #self.layer4 = self._make_layer(block, widths[3], layers[3], stride=2, dilation=1)
        # remove the final avgpool and fc layers

        # add extra layers
        #self.extra_layers = ExtraLayers(self.inplanes)
    
    
        # weight initialize
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0]*m.kernel_size[1]*m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        
    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None

        if stride != 1 or self.inplanes != planes*block.expansion:
            downsample = nn.Sequential(
                
                nn.Conv2d(self.inplanes, planes*block.expansion, kernel_size=1, stride=stride, dilation=dilation, bias=False),
                nn.BatchNorm2d(planes*block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample))
        self.inplanes = planes*block.expansion

        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
  
    
    def forward(self, x):
    
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        #out38x38 = x
        #x = self.layer3(x)
        #x = self.layer4(x)
        #out19x19 = x

        #out10x10, out5x5, out3x3, out1x1 = self.extra_layers(x)


        return x


# In[82]:


def resnet_base(depth, width=1, pretrained=True, **kwargs):

  # Construct a ResNet base network model for SSD


    if (depth not in [50, 101, 152]):
        raise ValueError('Choose 50, 101 or 152 for depth')

    if ((width != 1) and pretrained):
        raise ValueError('Does not support pretrained models with width > 1.')
    
  
    name_dict = {50: 'resnet50', 101: 'resnet101', 152: 'resnet152'}
    layers_dict = {50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3]}
    block_dict = {50: Bottleneck, 101: Bottleneck, 152: Bottleneck}
  
    model = ResNetBase(block_dict[depth], layers_dict[depth], width, **kwargs)
  
    if ((width == 1) and pretrained):
        pretrained_dict = model_zoo.load_url(model_urls[name_dict[depth]])
        model_state_dict = model.state_dict()
    
        # filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_state_dict}
        model_state_dict.update(pretrained_dict)
        model.load_state_dict(pretrained_dict)
        #for k in resnet_state_dict:
        #  print(k)
      
      
          #if k in model_state_dict.keys() and k.startwith('features'):
            #model_state_dict[k] = resnet_state_dict[k]
      
        #model.load_state_dict(model_state_dict)
        #model.load_state_dict(model_zoo.load_url(model_urls[name_dict[depth]]))
      
    return model 


# In[83]:


class L2Norm(nn.Module):
  
  # normalize all channels
    def __init__(self, in_features, scale):
        super(L2Norm, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features))
        self.reset_parameters(scale)
  
    def reset_parameters(self, scale):
        nn.init.constant_(self.weight, scale)
  
    def forward(self, x):
        x = F.normalize(x, dim=1)
        scale = self.weight[None, :, None, None]
        #print('Shape of scale: ',scale.shape)
        #print('Scale: ', scale)
        #print('Shape of x: ', x.shape)
        return scale*x


# In[84]:


class MultiBoxLayer(nn.Module):
  
    num_classes = 2   # conf_value + 1 background class
    num_anchors = [4,6,6,6,4,4]
    in_planes = [512, 1024, 512, 256, 256, 256]
  
  
    def __init__(self):
        super(MultiBoxLayer, self).__init__()

        #self.num_anchors = num_anchors
        self.loc_layers = nn.ModuleList()
        self.conf_layers = nn.ModuleList()

        for i in range(len(self.in_planes)):
            self.loc_layers.append(nn.Conv2d(self.in_planes[i], self.num_anchors[i]*4, kernel_size=3, padding=1))
            self.conf_layers.append(nn.Conv2d(self.in_planes[i], self.num_anchors[i]*self.num_classes, kernel_size=3, padding=1))
      
      
    def forward(self, xs):
        '''
         xs: (list) of tensor containing intermediate layer outputs.

         Returns:
              loc_preds: (tensor) predicted locations, sized [N,H*W*#anchors,4].
              conf_preds: (tensor) predicted class confidences, sized [N,8732,2].
        '''
    
        y_locs = []
        y_confs = []
        
        for i, x in enumerate(xs):
            y_loc = self.loc_layers[i](x)
            N = y_loc.size(0)
            y_loc = y_loc.permute(0,2,3,1).contiguous()
            y_loc = y_loc.view(N,-1,4)
            y_locs.append(y_loc)

            y_conf = self.conf_layers[i](x)
            y_conf = y_conf.permute(0,2,3,1).contiguous()
            y_conf = y_conf.view(N,-1,2)
            y_confs.append(y_conf)

        loc_preds = torch.cat(y_locs, 1)
        conf_preds = torch.cat(y_confs, 1)

        return loc_preds, conf_preds
  


# In[85]:


class SSD(nn.Module):
    input_size = 300          # change input size
  
    #num_classes = 2   # conf_value + 1 background class
    num_anchors = [4,6,6,6,4,4]
    in_planes = [512, 1024, 512, 256, 256, 256]
  
  
    def __init__(self, depth, width=1):
        super(SSD, self).__init__()

        self.base_network = resnet_base(depth, width)
        #print('resnet model_dict', self.base_network.state_dict().keys())
        self.norm4 = L2Norm(512, 2)        # size 38                                      # feature map size? 

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1)      # Error : Calculated padded input size per channel: (2 x 2). 
                                                                                       #Kernel size: (3 x 3). Kernel size can't be greater than actual input size
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1)

        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)

        self.conv7 = nn.Conv2d(1024, 1024, kernel_size=1)      # size 19

        self.conv8_1 = nn.Conv2d(1024, 256, kernel_size=1)
        self.conv8_2 = nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2)  #10

        self.conv9_1 = nn.Conv2d(512, 128, kernel_size=1)
        self.conv9_2 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2)  # 5

        self.conv10_1 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv10_2 = nn.Conv2d(128, 256, kernel_size=3)

        self.conv11_1 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv11_2 = nn.Conv2d(128, 256, kernel_size=3)


        self.multibox = MultiBoxLayer()
    
    
    def forward(self, x):
        hs = []

        h = self.base_network(x)
        #print('Shape of output after resnet: ', h.shape)
        hs.append(self.norm4(h))    # conv4_3     size 38
        #hs.append(h)
        h = F.max_pool2d(h, kernel_size=2, stride=2, ceil_mode=True)

        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h))
        h = F.max_pool2d(h, kernel_size=3, padding=1, stride=1, ceil_mode=True)

        h = F.relu(self.conv6(h))
        h = F.relu(self.conv7(h))
        hs.append(h)     # size 19

        h = F.relu(self.conv8_1(h))
        h = F.relu(self.conv8_2(h))
        hs.append(h)

        h = F.relu(self.conv9_1(h))
        h = F.relu(self.conv9_2(h))
        hs.append(h)

        h = F.relu(self.conv10_1(h))
        h = F.relu(self.conv10_2(h))
        hs.append(h)

        h = F.relu(self.conv11_1(h))
        h = F.relu(self.conv11_2(h))
        hs.append(h)

        loc_preds, conf_preds = self.multibox(hs)

        return loc_preds, conf_preds


# In[86]:


def test():
    model = SSD(depth=50, width=1)
    loc_preds, conf_preds = model(Variable(torch.randn(1,3,300,300)))
    print('Size of loc_preds: ', loc_preds.size())
    print('Size of conf_preds: ', conf_preds.size())


# In[87]:


test()


# In[88]:


get_ipython().system('jupyter nbconvert --to script ssd.ipynb')

