
import numpy as np
import torch
from torch.autograd import Variable
import os
import glob
import torch.nn as nn
import sys
import argparse

from sklearn.metrics.pairwise import cosine_similarity
import torchvision.models  as models
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
import time

import backbone
from saliency_pgd import *
from io_utils import model_dict, parse_args, get_resume_file ,get_assigned_file
from torch.utils.data.sampler import Sampler
from scipy import stats

print("process id ", os.getpid())


class TransformsC10:
    '''
    Apply the same input transform twice, with independent randomness.
    '''

    def __init__(self):
       
        # image augmentation functions
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    
        self.train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(.25,.25,.25),
            transforms.RandomRotation(2),
            transforms.ToTensor(),
        ])
        
        self.test_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor()
        ])
        

    def __call__(self, inp):
        out1 = self.train_transform(inp)
        return out1



parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default = "cifar" , help="increase output verbosity")
args = parser.parse_args()

    
image_size = 32
train_transform = TransformsC10() 

test_transform = train_transform.test_transform

testset = torchvision.datasets.CIFAR10(root='./dataset/', train=False,
                                           download=True, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                             shuffle=False, num_workers=8)


class WrappedModel(nn.Module):
    def __init__(self, module):
        super(WrappedModel, self).__init__()
        self.module = module # that I actually define.
    def forward(self, x):
        return self.module(x)



model = backbone.WideResNet28_10( flatten = True, beta_value = 50.)
checkpoint_dir = './checkpoints/%s/%s_%s_%s' %('cifar', 'WideResNet28_10', 'art' ,  'cifar')
model = WrappedModel(model)


    
print("resuming" , checkpoint_dir)
resume_file = get_resume_file(checkpoint_dir)
if resume_file is not None:
    print("resume_file" , resume_file)
    tmp = torch.load(resume_file)
    model.load_state_dict(tmp['state'])
else:
    print("error no file found")
    exit()

    
model = model.cuda()
model.eval()

def normalize(x):
    mean = torch.tensor([0.4914, 0.4822, 0.4465])
    std = torch.tensor([0.2023, 0.1994, 0.2010])
    new_std = std[..., None, None]
    new_mean = mean[..., None, None]
    return (x - new_mean.cuda())/new_std.cuda()
   

###single grad 
def get_grad(img, y, model):
    f , scores  = model.forward( {0:img , 1: 0})
    y = Variable(y.cuda())
    model.zero_grad()
    loss = lossfn(scores, y)
    g = torch.autograd.grad(loss, img)[0]
    gradient = g.detach()
    return gradient.cpu().numpy()

###integrated grad 
def get_gradIG(img ,y,model):
    steps = 100
    inputs_sq = img.detach().squeeze().cpu().numpy() 
    baseline = 0 * inputs_sq
    scaled_inputs = torch.from_numpy( np.array([baseline + (float(i) / steps) * (inputs_sq - baseline) for i in range(0, steps + 1)]))
    scaled_inputs = Variable(scaled_inputs).cuda()
    scaled_inputs.requires_grad = True
    _,scores = model({0:normalize(scaled_inputs),1:0})
    loss = lossfn(scores, y.repeat(steps+1).cuda())
    model.zero_grad()
    gradient = torch.autograd.grad(loss, scaled_inputs )[0]
    gradient = gradient.detach()
    
    avg_grads = torch.div(gradient[1:] + gradient[:-1] ,2.0)
    avg_grads = torch.mean(avg_grads, dim=0)
    integrated_grad = (inputs_sq - baseline) * avg_grads.unsqueeze(0).cpu().numpy()
    return integrated_grad
    


config1 = { 'epsilon': 8.0 / 255.0,
        'num_steps': 50,
        'step_size': 1./ 255.0,
        'k_top': 100,
        'img_size' : image_size,
        'num_ig_steps': 100
    }

attack1 = AttackSaliency(config1,normalize) 

print(config1)


lossfn = nn.CrossEntropyLoss()
total = 0.
correct = 0.
correct_adv = 0.
tau_values = [[],[]]
eval_k_top = 100

count = 0
for i , (x,y) in enumerate(testloader):
    y = Variable(y.cuda())
    x_var = Variable(x.cuda())
    _ , logits = model( {0:normalize(x_var) , 1:0} )
    p = torch.argmax(logits,1)
    correct_curr = (p==y).sum().item()
    if correct_curr < 1:
        continue
        
    xadv = attack1.saliency_attackIG(model, x_var , y , 0. , 1. )

    if xadv is None:
        continue
    
    count +=1
    _ , logits_adv = model( {0:normalize(xadv),1:0} ) 
    p1 = torch.argmax(logits_adv,1)
    correct += (p==y).sum().item()
    correct_adv += (p1==y).sum().item()
    total += p1.size(0)
    
    x_var.requires_grad = True
    xadv.requires_grad = True
    
    gg1_ig = get_gradIG(x_var ,y ,model)
    gg2_ig = get_gradIG(xadv ,y ,model)
   
    for each1 , each2 , x_each in zip(gg1_ig , gg2_ig , x_var):
        each1 = np.abs(each1).mean(0)
        each2 = np.abs(each2).mean(0)
        each1 = each1.flatten()
        each2 = each2.flatten()

        each1 = image_size*image_size * np.divide(each1, each1.sum())
        each2 = image_size*image_size * np.divide(each2, each2.sum())

        tau1 , v1 = stats.kendalltau( each1 , each2 )

        tau_values[0].append(tau1)   
        
        x_new = torch.abs(normalize(x_each).detach()).mean(0).cpu().numpy().flatten().reshape(1, -1)
       
        origin_ig_topK = np.argsort(each1)[-eval_k_top:]
        perturbed_ig_topK = np.argsort(each2)[-eval_k_top:]
        intersection = float(len(np.intersect1d(origin_ig_topK, perturbed_ig_topK))) / eval_k_top
        tau_values[1].append(intersection)

    if i%10==0:
        print(count , correct/total , correct_adv/total)
       
        print("kendall_ig" , np.mean(tau_values[0]))
        print("intersection_ig" , np.mean(tau_values[1]))
        
    if count > 1000:
        break

np.save('cifar_srt_eval.npy' , tau_values)

print(correct/total , correct_adv/total)

print("kendall_ig" , np.mean(tau_values[0]))
print("intersection_ig" , np.mean(tau_values[1]))