import math
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

def gaussian(window_size, sigma): # context of maximum-a-posteriori (MAP) estimation likely hood function signam is varience
	gauss = torch.Tensor([exp(-(x - window_size/2)**2/float(2*sigma**2)) for x in range(window_size)])#i think this is gauss dis function 
	#A torch.Tensor is a multi-dimensional matrix containing elements of a single data type.
	#https://stackoverflow.com/questions/57237352/what-does-unsqueeze-do-in-pytorch
	return gauss/gauss.sum()

def create_window(window_size, channel):
	_1D_window = gaussian(window_size, 1.5).unsqueeze(1) #Returns a new tensor with a dimension of size one inserted at the specified position.
	_2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)#Performs a matrix multiplication of the matrices input and mat2.
	#_1D_window.t() transpose  dimension is [1,1,4,4] if array is [1,2,3,4]
	#Args:
        #input (Tensor): the input tensor.
        #dim (int): the index at which to insert the singleton dimension
	window = Variable(_2D_window.expand(channel, 1, window_size, window_size))#A Variable wraps a Tensor. It supports nearly all the API’s defined by a Tensor. 
	#Variable also provides a backward method to perform backpropagation 
	# tensor,requires_grad,(vairiable)
	#expand The difference is that if the original dimension you want to expand is of size 1, you can use torch.expand() to do it without using extra memory.
	return window

def SSIM(img1, img2):# Structural Similarity Index more it is good the image deblurred 
	(_, channel, _, _) = img1.size() #https://en.wikipedia.org/wiki/Structural_similarity#:~:text=For%20an%20image%2C%20it%20is,quality%20map%20of%20the%20image.
        #ssim math
	window_size = 11
	window = create_window(window_size, channel).cuda()
	mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)#Applies a 2D convolution over an input image composed of several input planes.
	#split input into groups, \text{in\_channels}in_channels should be divisible by the number of groups.
	mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)
        #o/p is N,Cout,Hout,Wout :(batch size,channels,Height of the input planeof pixels,width of pixels)
	mu1_sq = mu1.pow(2)  #mu1=window1 mu2=window2
	mu2_sq = mu2.pow(2)
	mu1_mu2 = mu1*mu2

	sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
	sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
	sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

	C1 = 0.01**2 #see wiki link above
	C2 = 0.03**2

	ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
	return ssim_map.mean()

def PSNR(img1, img2, mask=None): #https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
	if mask is not None: #mask / kernel
		mse = (img1 - img2) ** 2
		B,C,H,W=mse.size() 
		mse = torch.sum(mse * mask.float()) / (torch.sum(mask.float())*C)
	else:
		mse = torch.mean( (img1 - img2) ** 2 ) #mse of image 
	
	if mse == 0: #if images are identical then 
		return 100
	PIXEL_MAX = 1 #formula 
	return 20 * math.log10(PIXEL_MAX / math.sqrt(mse)) 
