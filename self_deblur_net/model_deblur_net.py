import torch
import torch.nn as nn

from model_base import *
from net_deblur import *
from metrics import *

class ModelDeblurNet(ModelBase): #class model base in import for saving,loading etc 
    def __init__(self, opts): # this opts is in inference.py use for congiuration of traning settings
        super(ModelDeblurNet, self).__init__() #super is good for multiple inhertance and avoid referring the base class 
        self.opts = opts  #attribute for congirutation of training settings 
        
        # create network
        self.model_names=['G'] #giving model a name 
        self.net_G=Deblur_net(n_in=opts.n_channels, n_init=opts.n_init_feat, n_out=opts.n_channels).cuda() #.cuda GPU access to deblur net 

        self.print_networks(self.net_G) #print the NN 

        if not opts.is_training or opts.continue_train: #if we are not training or continuing training then 
            self.load_checkpoint(opts.model_label) #load the weights to model

        if opts.is_training: # if the model is training then 
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.net_G.parameters(), lr=opts.lr) #use adam with lr=1e-4 opts.lr is train_deblur_net.py even though it is not imported 
            # how is it accessing opts.lr ?
            self.optimizer_names = ['G'] # this part is used in lr scheduler in model_base.py
            self.build_lr_scheduler() # is in model_base.py 
            
            self.loss_fn=nn.L1Loss() #l1 loss in losses.py
                        
    def set_input(self, _input):
        im_blur, im_target = _input
        self.im_blur=im_blur.cuda() #blur image into gpu 
        if im_target is not None: # if there is  truth image here it is the blur image only
            self.im_target=im_target.cuda() # take it into gpu 
        
    def forward(self):
        im_pred=self.net_G(self.im_blur)# into the deblur net send the blur image 
        return im_pred  # return the predicted image 

    def optimize_parameters(self):
        self.im_pred=self.forward() #store the prediction 

        self.loss_G=self.loss_fn(self.im_pred, self.im_target)# calculate loss by taking blur and target 

        self.optimizer_G.zero_grad() #to avoid over writing of weights when back prop
        self.loss_G.backward() # i think back prop computes dloss/dx for every parameter x
        self.optimizer_G.step()  # https://discuss.pytorch.org/t/how-are-optimizer-step-and-loss-backward-related/7350 
        # optimizer.step is performs a parameter update based on the current gradient

    def save_checkpoint(self, label): # this is in model_base 
        self.save_network(self.net_G, 'deblur', label, self.opts.log_dir)
        
    def load_checkpoint(self, label):
        self.load_network(self.net_G, 'deblur', label, self.opts.log_dir)

    def get_current_scalars(self): #get train info like loss,psnr 
        losses = {}
        losses['loss_G']=self.loss_G.item()
        losses['PSNR_train']=PSNR(self.im_pred.data, self.im_target)
        return losses

    def get_current_visuals(self):
        output_visuals = {}
        output_visuals['im_blur']=self.im_blur
        output_visuals['im_target']=self.im_target
        output_visuals['im_pred']=self.im_pred
        return output_visuals
