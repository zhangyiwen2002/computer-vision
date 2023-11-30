#coding=utf-8
import sys,os
sys.path.append('opencv/python')
import torch,cv2,random,time,datetime
import numpy as np
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

class cnnNet(torch.nn.Module):
    def __init__(self):
        super(cnnNet,self).__init__()
        '''
        #无padding
        #第一层卷积神经网络
        #输入通道定义= 1 ，输出通道维度32 卷积核大小3*3
        self.conv1 = torch.nn.Conv2d(3,32,3)
        #定义第二层卷积神经网络
        #输入通道维度=32 ，输出通道维度=64 ，卷积核大小3*3
        self.conv2 = torch.nn.Conv2d(32,64,3)
        #定义三层全连接层,卷积神经网络中至少有一个全连接层，是将卷积层
        self.fc1 = torch.nn.Linear(64*6*6,128) #第一层的核大小是前一层卷积层的输出核大小64*,256是隐变量大小
        self.fc2 = torch.nn.Linear(128,10)  #10 分类
        '''
        
        #带padding
        #第一层卷积神经网络
        #输入通道定义= 3 ，输出通道维度32 卷积核大小3*3
        self.conv1 = torch.nn.Conv2d(in_channels=3,out_channels=32,kernel_size=(3,3),stride=(1,1),padding=1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        #定义第二层卷积神经网络
        #输入通道维度=32 ，输出通道维度=64 ，卷积核大小3*3
        self.conv2 = torch.nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(3,3),stride=(1,1),padding=1)
        self.bn2 = torch.nn.BatchNorm2d(64)
        #定义三层全连接层,卷积神经网络中至少有一个全连接层，是将卷积层
        self.fc1 = torch.nn.Linear(64*8*8,10) #第一层的核大小是前一层卷积层的输出核大小64*,256是隐变量大小
        #self.fc2 = torch.nn.Linear(128,10)  #10 分类
        
    def forward(self,x):
        #卷积层 后面加激活层核池化层
        #print(np.shape(x))
        x = self.conv1(x)
        bn  = self.bn1(x)
        x = F.max_pool2d(F.relu(bn),(2,2))#池化层
        #print(np.shape(x))
        x = self.conv2(x)
        bn  = self.bn2(x)
        x = F.max_pool2d(F.relu(bn),(2,2))
        #print(np.shape(x))
        #经过卷积层的处理后，张量需要调整，进入前需要调整张量的形状
        x = x.view(-1,self.num_flat_features(x)) #这里直接写下面方法也可以
        #x = x.view(-1,64 * 6 * 6))
        # 激活层
        
        x = self.fc1(x)
        #x = F.relu(x)
        #x = self.fc2(x)
        x = F.softmax(x,dim=1)
        return x
        
    def num_flat_features(self,x):
        #除了第0 维度的batch_size
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

if __name__=='__main__':
    scaleSize = 32
    numClass = 108
    #device=torch.device("cuda:0")
    device=torch.device("cuda:0")
    net = cnnNet()    #初始化网络
    net.to(device)    #将网络推到CPU内存或者GPU显存
    net.eval()       #不更新bn的参数
    
    modelPath = './models/ourNet-best.pth'
    #载入模型，修改类别
    netPre = torch.load(modelPath)
    net.load_state_dict(netPre, strict=False)
    
    net.to(device)
    #net.train()      #更新bn的参数
    net.eval()        #不更新bn的参数
    
    imgpath = './mnist/test/1/3.jpg'   #举例一张测试图片
    img = cv2.imread(imgpath)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = Image.fromarray(img)
        
    aug = transforms.Compose([   
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.3081, 0.3081, 0.3081]), # 正规化
                transforms.Resize([scaleSize, scaleSize])
            ])
    
    img = aug(img)
    imgBatch = torch.randn(1,3,scaleSize,scaleSize)
    imgBatch[0,:,:,:] = img

    inputs = imgBatch.to(device)
    outputs = net(inputs)
    _, predicted = outputs.max(1)
    predicted = np.array(predicted.detach().cpu().numpy())
    print('预测的类别：',predicted[0])
    