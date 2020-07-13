import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

class AttackPGD(nn.Module):
    def __init__(self,  config , normalize):
        super(AttackPGD, self).__init__()
        self.step_size = config['step_size']
        self.eps = config['epsilon']
        self.num_steps = config['num_steps']
        self.criterion = F.cross_entropy
        self.normalize_fn = normalize


    def attack(self, net, images, labels , minim , maxim):
        adv = images.clone()
        with torch.no_grad():
            adv = adv + 2* self.eps * (torch.rand(images.size()).cuda() - 0.5)
            adv = torch.clamp(adv , minim , maxim)
            
        for i in range(self.num_steps):
            adv.requires_grad = True

            if adv.grad is not None:
                adv.grad.data._zero()
            
            _ , outputs = net({0:self.normalize_fn(adv) ,1:0 }) 
            loss = self.criterion(outputs, labels)
            loss.backward()

            with torch.no_grad():
                adv = adv + self.step_size * torch.sign(adv.grad)
                adv = torch.min(adv, images+self.eps)
                adv = torch.max(adv, images-self.eps)
                adv = torch.clamp(adv, minim,  maxim)  
        
        return adv