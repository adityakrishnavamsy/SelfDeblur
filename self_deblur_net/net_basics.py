import torch
import torch.nn as nn

## 2D convolution layers
class Conv2d(nn.Module):
    def __init__(self, n_in, n_out, bn, act_fn, ker_sz=3, strd=1):
        super(Conv2d, self).__init__() #https://www.kite.com/python/answers/how-to-use-super()-in-python 
        # super is used for additional functionality to intilialize the conv2d class here to reduce the duplication 
        
        use_bias=True if bn==False else False #bn batch normalization 
        #The bias is an additive parameter in the convolution. It’s like the b in f(x) = w*x + b. If you set bias=False, you will drop the b term, which might make
        #sense in some cases, e.g. if the next layer is an affine BatchNorm layer
         
        modules = []   #Modules in Python are simply Python files with a .py extension. The name of the module will be the name of the file
        modules.append(nn.Conv2d(n_in, n_out, ker_sz, stride=strd, padding=(ker_sz-1)//2, bias=use_bias)) 

        if bn==True:
            modules.append(nn.BatchNorm2d(n_out))# number of features=nout
            #Batch normalization layers normalize the activations and gradients propagating through a neural network,
            #making network training an easier optimization problem
        
        if act_fn is not None: #if there are  activation function 
            modules.append(act_fn)

        self.net=nn.Sequential(*modules) #i think layers is *modules and it is a sequential model 

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d): #The isinstance() function returns True if the specified object is of the specified type, otherwise False i.e is element in module a CNN
                nn.init.xavier_uniform_(m.weight)#https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch
                #use to initialize weights of a layer 
                if m.bias is not None:# if there is bias in that layer 
                    m.bias.data.zero_() #set bais to 0 ?
    
    def forward(self, x):
        return self.net(x) # forward NN sequential 

class Deconv2d(nn.Module):
    def __init__(self, n_in, n_out):
        super(Deconv2d, self).__init__() #super() lets you avoid referring to the base class explicitly, 
        #which can be nice. But the main advantage comes with multiple inheritance # https://stackoverflow.com/questions/576169/understanding-python-super-with-init-methods
      
        self.upsample=nn.Upsample(scale_factor=2, mode="bilinear")#Upsampling layer is a simple layer with no weights that will double the dimensions of input and 
        #can be used in a generative model when followed by a traditional convolutional layer
        self.net=nn.Sequential(Conv2d(n_in, n_out, False, None, 3, 1)) #these are classes so it is a seqclass and conv2d class agr are needed 

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):#Applies a 2D transposed convolution operator over an input image composed of several input planes
                #Transpose is learning parameter while Up-sampling is no-learning parameters.
                # ? why they are using this when module array has only Conv2d,activation,bn some times 
                #Using Up-samling for faster inference or training because it does not require to update weight or compute gradient
                # https://discuss.pytorch.org/t/torch-nn-convtranspose2d-vs-torch-nn-upsample/30574
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):# forward nn
        return self.net(self.upsample(x))

class Resnet_block(nn.Module):
    def __init__(self, n_in):
        super(Resnet_block, self).__init__()#making it a super class 

        self.conv_block = self.build_conv_block(n_in)#see down there is a function/method 

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):#if m is of type nn.Conv2d
                nn.init.xavier_uniform_(m.weight)#the initialize weights 
                if m.bias is not None:#if there is baisa make it to 0
                    m.bias.data.zero_()

    def build_conv_block(self, n_in):#takes input featues i think 
        conv_block=[]
        conv_block+=[Conv2d(n_in, n_in, False, nn.ReLU(), 3, 1)]#kernel=3 stride=1,bn(batch normalization)=false
        conv_block+=[Conv2d(n_in, n_in, False, None, 3, 1)]
        return nn.Sequential(*conv_block)#a sequential model with layers are in conv_block

    def forward(self, x): # x is i think n_in
        return x+self.conv_block(x)

class Cascaded_resnet_blocks(nn.Module):
    def __init__(self, n_in, n_blks):#n_in=number of input features #n_blks=number of blocks 
        super(Cascaded_resnet_blocks, self).__init__()

        resnet_blocks=[]
        for i in range(n_blks):#for number of blocks 
            resnet_blocks+=[Resnet_block(n_in)]#see there is a method in for it
        self.net = nn.Sequential(*resnet_blocks)#seq model 

        # initialization
        for m in self.modules(): # ? why they are using this when module array has only Conv2d,activation,bn some times 
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)#if Conv2d then initialize weights with xvavier             
                if m.bias is not None: #if there is bais in that layer make it 0
                    m.bias.data.zero_()

    def forward(self, x):
        return self.net(x)
