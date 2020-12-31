import cv2
import torch
import numpy as np
import torch.nn as nn

def white_balance(img): #An important goal of this adjustment is to render specific colors – particularly neutral colors – correctly
    img = (img*255.).astype(np.uint8) #https://en.wikipedia.org/wiki/Color_balance
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB) #https://stackoverflow.com/questions/54470148/white-balance-a-photo-from-a-known-point
    avg_a = np.average(img[:, :, 1])
    avg_b = np.average(img[:, :, 2])
    img[:, :, 1] = img[:, :, 1] - ((avg_a - 128) * (img[:, :, 0] / 255.0) * 1.1)
    img[:, :, 2] = img[:, :, 2] - ((avg_b - 128) * (img[:, :, 0] / 255.0) * 1.1)
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    img = img.astype(np.float)/255.
    return img

def warp_image_flow(ref_image, flow):  # here ref image and flow both are o/p of a network so they are differentiable 
    [B, _, H, W] = ref_image.size() #B= i think batch , H=height , W=width see 4/8 base paper  [batch_size, channels, height, width] 
    
    # mesh grid 
    xx = torch.arange(0, W).view(1,-1).repeat(H,1) # making a grid of the referecne image 
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()

    if ref_image.is_cuda: # GPU 
        grid = grid.cuda()

    flow_f = flow + grid  #warping each vertex of the lattice wrt the optical flow 
    flow_fx = flow_f[:, 0, :, :] 
    flow_fy = flow_f[:, 1, :, :]

    with torch.no_grad(): #will make all the operations in the block have no gradients.
        #https://datascience.stackexchange.com/questions/32651/what-is-the-use-of-torch-no-grad-in-pytorch
        mask_x = ~((flow_fx < 0) | (flow_fx > (W - 1))) #mask is used to deal with oclussion 5/8 pg 
        mask_y = ~((flow_fy < 0) | (flow_fy > (H - 1))) #when ever there is a ocluded region mask is applied 
        mask = mask_x & mask_y  
        mask = mask.unsqueeze(1) #Returns a new tensor with a dimension of size one inserted at the specified position
        #https://pytorch.org/docs/stable/generated/torch.unsqueeze.html

    flow_fx = flow_fx / float(W) * 2. - 1.
    flow_fy = flow_fy / float(H) * 2. - 1.

    flow_fxy = torch.stack([flow_fx, flow_fy], dim=-1)
    img = torch.nn.functional.grid_sample(ref_image, flow_fxy, padding_mode='zeros') 
    #iven an input and a flow-field grid, computes the output using input values and pixel locations from grid
    #https://pytorch.org/docs/stable/nn.functional.html
    return img, mask

class Grid_gradient_central_diff():
    def __init__(self, nc, padding=True, diagonal=False):
        self.conv_x = nn.Conv2d(nc, nc, kernel_size=2, stride=1, bias=False)
        self.conv_y = nn.Conv2d(nc, nc, kernel_size=2, stride=1, bias=False)
        self.conv_xy = None
        if diagonal:
            self.conv_xy = nn.Conv2d(nc, nc, kernel_size=2, stride=1, bias=False)
    
        self.padding=None
        if padding:
            self.padding = nn.ReplicationPad2d([0,1,0,1])

        fx = torch.zeros(nc, nc, 2, 2).float().cuda()
        fy = torch.zeros(nc, nc, 2, 2).float().cuda()
        if diagonal:
            fxy = torch.zeros(nc, nc, 2, 2).float().cuda()
        
        fx_ = torch.tensor([[1,-1],[0,0]]).cuda()
        fy_ = torch.tensor([[1,0],[-1,0]]).cuda()
        if diagonal:
            fxy_ = torch.tensor([[1,0],[0,-1]]).cuda()

        for i in range(nc):
            fx[i, i, :, :] = fx_
            fy[i, i, :, :] = fy_
            if diagonal:
                fxy[i,i,:,:] = fxy_
            
        self.conv_x.weight = nn.Parameter(fx)
        self.conv_y.weight = nn.Parameter(fy)
        if diagonal:
            self.conv_xy.weight = nn.Parameter(fxy)

    def __call__(self, grid_2d):
        _image = grid_2d
        if self.padding is not None:
            _image = self.padding(_image)
        dx = self.conv_x(_image)
        dy = self.conv_y(_image)

        if self.conv_xy is not None:
            dxy = self.conv_xy(_image)
            return dx, dy, dxy
        return dx, dy
