#coding=utf-8
import sys,os
from resnet import *
sys.path.append('opencv/python')
import torch,cv2,random,time,datetime
import numpy as np
from queue import Queue
from threading import Thread
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

# 图片预处理   
def pre_process_batch(trainlist,randomIndex,trainLabel,batch):
    scaleSize = 32
    imgBatch = torch.randn(batch,3,scaleSize,scaleSize)
    i = 0
    for imgpath in trainlist[randomIndex]:
        img = cv2.imread(imgpath)
        img = cv2.resize(img, (scaleSize, scaleSize))
        img = Image.fromarray(img)
        aug = transforms.Compose([   
                #transforms.RandomRotation(5), # 随机旋转
                #transforms.GaussianBlur(kernel_size=3, sigma=(1.0, 3.0)), # 高斯模糊
                transforms.RandomCrop(scaleSize, padding=4), # 随机裁剪
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.3081, 0.3081, 0.3081]), # 正规化
                #transforms.RandomErasing(p=1.0, scale=(0.005, 0.005), ratio=(0.3, 3.3)), # 随机擦除
            ])
        '''
        aug = transforms.Compose([   
                transforms.RandomRotation(10), # 随机旋转
                transforms.ColorJitter(brightness=0.2,contrast=0.2, saturation=0.05, hue=0.05), # 亮度变换
                transforms.GaussianBlur(kernel_size=3, sigma=(1.0, 3.0)), # 高斯模糊
                transforms.RandomCrop(scaleSize, padding=4), # 随机裁剪
                transforms.RandomHorizontalFlip(), # 水平翻转
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.1307, 0.1307, 0.1307], std=[0.3081, 0.1307, 0.1307]), # 正规化
                transforms.RandomErasing(p=1.0, scale=(0.005, 0.005), ratio=(0.3, 3.3)), # 随机擦除
            ])
        '''
        img = aug(img)
        imgBatch[i,:,:,:] = img
        i = i+1
    labelBatch =torch.LongTensor(np.array(trainLabel[randomIndex],dtype=int))
    imgAndLabel = dict()
    imgAndLabel['images'] = imgBatch
    imgAndLabel['label'] = labelBatch

    return imgAndLabel
    
def load_images(trainlist,trainLabel,batch,images_queue):
    
    while True:
        randomIndex = random.sample(range(0,len(trainlist)),batch)
        imgAndLabel = pre_process_batch(trainlist,randomIndex,trainLabel,batch)
        images_queue.put(imgAndLabel)
        
def load_test_images(testlist,testLabel,batch,k):
    batchIndex = np.arange(0,batch,1)+k*batch
    imgAndLabel = pre_process_batch(testlist,batchIndex,testLabel,batch)
    return imgAndLabel
        
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
        
def trainNet(device,num_epoch,batch_epoch,testlist,testLabel,images_queue):
    net = cnnNet()    #初始化网络
    net.to(device)    #将网络推到CPU内存或者GPU显存
    net.train()       #更新bn的参数
    #net.eval()       #不更新bn的参数

    loss_fn = torch.nn.CrossEntropyLoss()    #定义loss函数
    
    #optimizer = torch.optim.RMSprop(net.parameters(),learningRate)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=.9, weight_decay=5e-4)    #梯度下降法学习率设置
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)               #梯度下降寻优方案
    loss_list = []
    val_acc_list = []
    lossBest = 100000
    print('Start train!')
    for epoch in range(num_epoch): 
        train_loss = 0 
        
        for j in range(batch_epoch):     
                        
            imgAndLabel = images_queue.get()                #从images_queue队列中获取一个批次的图像和标签
            imageBatch = imgAndLabel['images']              #一个批次的图像
            labelBatch = imgAndLabel['label']               #一个批次的标签
            inputs,labels = imageBatch.to(device), labelBatch.to(device)      #将批次图像推到设备（CPU内存或者GPU显存）
            optimizer.zero_grad()                           #清空历史梯度
            outputs = net(inputs)                           #网络的前向传播
            loss = loss_fn(outputs,labels)                  #利用标签计算loss
            train_loss += loss.item()                       #loss累加
            loss.backward()                                 #反向传播，计算梯度
            optimizer.step()                                #更新网络权重参数

        scheduler.step()                                    #学习率测量调整
        print('images_queue size:',images_queue.qsize(),'   ',epoch,' epoch ','loss:',train_loss/(j+1))
        
        if epoch%2==0:                                     #每训练10批次图像，检验一下在测试集的准确率
            net.eval()                                      #测试过程，关闭BN参数更新模式
            
            # 计算验证集精度
            correct = 0
            total = 0
            test_loss = 0
            for j in range(int(len(testlist)/batch)):
                imgAndLabel = load_test_images(testlist,testLabel,batch,j)        #载入一个批次的测试数据集
                imageBatch = imgAndLabel['images']                                #一个批次的图像
                labelBatch = imgAndLabel['label']                                 #一个批次的标签
                inputs,labels = imageBatch.to(device), labelBatch.to(device)      #将批次图像推到设备（CPU内存或者GPU显存）
                outputs = net(inputs)                                             #网络的前向传播
                _, predicted = outputs.max(1)                                     #选择输出结果最大的一类
                total += labelBatch.size(0)                                       #累计测试的图片数量
                correct += predicted.eq(labels).sum().item()                      #累计测试正确率的数量

                loss = loss_fn(outputs,labels)                                    #利用标签计算测试的loss
                test_loss += loss.item()                                          #测试loss累加

            acc = 100.*correct/total                                              #计算测试的正确率
            val_acc_list.append(acc)                                              #将测试保存到一个变量中
            loss_list.append(test_loss/(j+1))                                     #将测试loss保存到一个变量中
            
            print(epoch,'-epoch loss:',test_loss/(j+1),'   Test acc:',acc)        #打印测试数据集的结果
            
            if test_loss/(j+1)<lossBest:
                lossBest = test_loss/(j+1)
                torch.save(net.state_dict(),'./models/ourNet-best.pth')           #保存最佳模型
                
            torch.save(net.state_dict(),'./models/ourNet-last.pth')               #保存模型
            net.train()                                                           #将模型切换到回BN参数更新模式
    torch.save(net.state_dict(),'./models/ourNet-last.pth')                       #保存最后的模型
          
if __name__=='__main__':
    #载入一个batch的数据
    batch = 32
    num_epoch = 200
    device=torch.device("cuda:0")          #在GPU设备上训练
    #device=torch.device("cpu")              #在cpu设备上训练
    images_queue = Queue(maxsize=20)        #训练数据的队列
    test_images_queue = Queue(maxsize=20)   #测试数据的队列
    
    trainlistPath  = './mnist/trainSamples.txt'
    trainLabelPath = './mnist/trainLabels.txt'
    testlistPath   = './mnist/testSamples.txt'
    testLabelPath  = './mnist/testLabels.txt'

    #运行载入数据线程
    trainlist = np.loadtxt(trainlistPath,dtype=str)      #载入训练数据集的所有图片路径
    trainLabel = np.loadtxt(trainLabelPath,dtype=str)    #载入训练数据集的对应标签
    testlist = np.loadtxt(testlistPath,dtype=str)        #载入测试数据集的所有图片路径
    testLabel = np.loadtxt(testLabelPath,dtype=str)      #载入测试数据集的对应标签
    
    batch_epoch = int(len(trainlist)/batch)              #一轮的batch数量
    
    Thread(target=load_images, args=(trainlist, trainLabel,batch,images_queue)).start()                          #启动图片载入线程
    Thread(target=trainNet, args=(device,num_epoch,batch_epoch,testlist,testLabel,images_queue)).start()         #启动训练线程
    