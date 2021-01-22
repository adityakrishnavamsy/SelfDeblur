import torch
import os
import numpy as np
from torch.optim import lr_scheduler    #learning rate scheduler as optimizer 

class ModelBase():
    def save_network(self, network, network_label, epoch_label, save_dir, on_gpu=True): #save model 
        if not os.path.exists(save_dir): #if the file is not saved then create a directory 
            os.makedirs(save_dir)
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label) #path files extension 
        save_path = os.path.join(save_dir, save_filename) #https://www.geeksforgeeks.org/python-os-path-join-method/ joins paths 
        torch.save(network.cpu().state_dict(), save_path)#save or load entire model when CPU 

        if on_gpu:
            network.cuda() #if on GPU 

    def load_network(self, network, network_label, epoch_label, save_dir):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(save_dir, save_filename)  
        network.load_state_dict(torch.load(save_path))#laod network 
        print('load network from ', save_path)

    def print_networks(self, net):
        print('---------- Networks initialized -------------')
        num_params = 0
        for param in net.parameters(): #learnable parameters of a model are returned by
            num_params += param.numel()#number of parameters 
        print(net)
        print('[Network] Total number of parameters : %.3f M' % (num_params / 1e6))
        print('-----------------------------------------------')

    # used in test time, wrapping `forward` in no_grad() so we don't save
    # intermediate steps for backprop
    def eval(self):
        with torch.no_grad():#Disabling gradient calculation is useful for inference, when you are sure that you will not call Tensor.backward()
            return self.forward() #? used for connecting i think  

    def build_lr_scheduler(self):
        self.lr_schedulers = []
        for name in self.optimizer_names:
            if isinstance(name, str):#if name is of type string 
                optimizer = getattr(self, 'optimizer_' + name) #getattr() function is used to get the value of an object's attribute 
                self.lr_schedulers.append(lr_scheduler.StepLR(optimizer, step_size=self.opts.lr_step, gamma=0.5))#Decays the learning rate of  #self.opts.lr_step
                #each parameter group by gamma every step_size epochs.

    def update_lr(self):
        for scheduler in self.lr_schedulers: #update the learning for step
            scheduler.step()

        for name in self.optimizer_names:
            if isinstance(name, str):
                optimizer = getattr(self, 'optimizer_' + name)
                for param_group in optimizer.param_groups:
                    print('optimizer_'+name+'_lr', param_group['lr']) #i think to print the optiizer with schedulers 
