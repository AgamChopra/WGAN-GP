import torch
import torch.nn as nn
import numpy as np
import sys 
import gans1dset as dst
CH = 3

def normalize_imgs(x):
    return ((2*x)/255)-1.


def denormalize_imgs(x):
    return (x+1)*255./2
    

#Critic    
class Encoder2(nn.Module):
    def __init__(self):
        super(Encoder2, self).__init__()
        c1,c2,c3,c4 = 128, 256, 512, 1024
        self.l1 = nn.Sequential(nn.Conv2d(CH, c1, 4, 2, padding = 1), nn.InstanceNorm2d(c1,affine=True), nn.LeakyReLU(0.2,inplace=False),
                                nn.Conv2d(c1, c2, 4, 2, padding = 1), nn.InstanceNorm2d(c2,affine=True), nn.LeakyReLU(0.2,inplace=False),
                                nn.Conv2d(c2, c3, 4, 2, padding = 1), nn.InstanceNorm2d(c3,affine=True), nn.LeakyReLU(0.2,inplace=False),
                                nn.Conv2d(c3, c4, 4, 2, padding = 1), nn.InstanceNorm2d(c4,affine=True), nn.LeakyReLU(0.2,inplace=False))
        self.l2 = nn.Conv2d(c4, 1, 4, 1)
        
    def forward(self, x):
        y = self.l1(x)
        y = self.l2(y)
        return  y
    

#Generator
class AutoEnc(nn.Module):
    def __init__(self):
        super(AutoEnc, self).__init__()
        c1,c2,c3,c4 = 1024, 512, 250, 128
        self.E = nn.Sequential(nn.ConvTranspose2d(100, c1, 4, 1), nn.InstanceNorm2d(c1,affine=True), nn.LeakyReLU(0.2,inplace=False),
                               nn.ConvTranspose2d(c1, c2, 4, 2, padding = 1), nn.InstanceNorm2d(c2,affine=True), nn.LeakyReLU(0.2,inplace=False),
                               nn.ConvTranspose2d(c2, c3, 4, 2, padding = 1), nn.InstanceNorm2d(c3,affine=True), nn.LeakyReLU(0.2,inplace=False),
                               nn.ConvTranspose2d(c3, c4, 4, 2, padding = 1), nn.InstanceNorm2d(c4,affine=True), nn.LeakyReLU(0.2,inplace=False),
                               nn.ConvTranspose2d(c4, CH, 4, 2, padding = 1), nn.Tanh())        
    
    def forward(self, x):
        y = self.E(x)
        return  y


def grad_penalty(critic, real, fake, device='cpu'):
    b_size, c, h, w = real.shape
    epsilon = torch.rand(b_size, 1, 1, 1).repeat(1,c,h,w).to(device)
    interp_img = real * epsilon + fake * (1 - epsilon)
    
    mix_score = critic(normalize_imgs(interp_img))
    
    grad = torch.autograd.grad(outputs = mix_score, 
                                   inputs = interp_img, 
                                   grad_outputs=torch.ones_like(mix_score),
                                   create_graph=True,
                                   retain_graph=True)[0]
    
    grad = grad.view(grad.shape[0],-1)
    grad_norm = grad.norm(2,dim=1)
    penalty = torch.mean((grad_norm - 1) ** 2)
    return penalty


def advers_train(lr = 1E-4, epochs = 5, batch=32, beta1=0.5, beta2=0.999, critic_iter=5, Lambda_penalty = 10):
    
    Closses = []
    Glosses = []
    
    cat = dst.photo_dataset()
    cat = torch.from_numpy(cat).to(dtype = torch.float)
      
    #Generator 
    AutoE = AutoEnc().cuda()
    
    #Critic
    E2 = Encoder2().train().cuda()     
    
    optimizerG = torch.optim.Adam(AutoE.parameters(),lr,betas=(beta1, beta2))
    optimizerC = torch.optim.Adam(E2.parameters(),lr,betas=(beta1, beta2))
    
    for eps in range(epochs):    
        
        for b in range(0,cat.shape[0]-batch,batch): 
            
            idx_ = torch.randperm(cat.shape[0])
            ###Critic###
            for j in range(1, critic_iter + 1):
                real = cat[idx_[batch*j:batch*j+batch]].cuda()
                x = (torch.rand((real.shape[0],100,1,1)).cuda() * 2) - 1
                fake = AutoE(x)
                
                optimizerC.zero_grad()             
                #Real
                y = E2(normalize_imgs(real))
                errC_real = torch.mean(y.reshape(-1))
                
                #fake
                y = E2(normalize_imgs(fake.detach()))
                errC_fake = torch.mean(y.reshape(-1))
                
                #total err/loss
                penalty = grad_penalty(E2, real, fake, device='cuda')
                errC = - (errC_real - errC_fake) + Lambda_penalty * penalty                   
                errC.backward(retain_graph=True)
                optimizerC.step()
            
            
            ###Generator###
            x = (torch.rand((real.shape[0],100,1,1)).cuda() * 2) - 1
            fake = AutoE(normalize_imgs(x))
            
            optimizerG.zero_grad()            
            y = E2(normalize_imgs(fake))
            errG = -torch.mean(y.reshape(-1))
            errG.backward()            
            optimizerG.step() 
            
            
            #Misc.
            Closses.append(errC.item())
            Glosses.append(errG.item())  
            
            
            if b%(batch*125) == 0:
                print('[%d/%d][%d/%d]\tLoss_C: %.4f\tLoss_G: %.4f'% (eps, epochs, int(b/batch), int(len(idx_)/batch), errC.item() - penalty, errG.item()))
                dst.visualize(denormalize_imgs(fake[0:25]).cpu().detach().numpy().astype(np.uint8))               
        
        
    torch.save(AutoE.state_dict(), r"xyz.pth")
    torch.save(E2.state_dict(), r"zzz.pth")  
    
    
    return AutoE, E2, Closses, Glosses
#%%

AE,E2,Dl,Gl = advers_train(lr=1E-4,epochs=300,batch=64,critic_iter=5)#64

dst.plt.figure(figsize=(10,5))
dst.plt.title("Generator and Discriminator Loss During Training")
dst.plt.plot(Dl, label='D_loss')
dst.plt.plot(Gl, label='G_loss')
dst.plt.legend()
dst.plt.xlabel("iterations")
dst.plt.ylabel("Loss")
dst.plt.legend()
dst.plt.show()
#%%
AE = AutoEnc().cuda()
AE.load_state_dict(torch.load(r"xyz.pth"))
noise = torch.rand((25,100,1,1)).cuda()
warped = AE(noise)
wd = warped.cpu().detach().numpy()
print(wd[1].shape)
dst.visualize(denormalize_imgs(wd).astype(np.uint8))
dst.visualize(denormalize_imgs(wd[0]).astype(np.uint8))
