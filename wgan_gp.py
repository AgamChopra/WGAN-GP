'''
My implementation of a Wasserstein Generative Adversarial 
Network(WGAN) with Gradient Penelty(-GP) for Lipschitz constraint 
and generator added noise to real data for improved Critic stability.
Resources:
    https://arxiv.org/abs/1701.07875
    https://arxiv.org/abs/1704.00028
'''
import torch
import torch.nn as nn
import numpy as np
import gans1dset as dst #Change this with your own dataset lib. and modify data loading and visualization accordingly.


CH = 3 #Expected color channel depth.
EPOCHS = 500

#Utilities

def normalize_imgs(x):
    return ((2*x)/255)-1


def denormalize_imgs(x):
    return (x+1)*255/2
    

#Critic Network   
class Encoder2(nn.Module): # (N,3,128,128) -> (N,1)
    def __init__(self):
        super(Encoder2, self).__init__()
        c1,c2,c3,c4 = 256, 512, 1024, 2048
        self.l1 = nn.Sequential(nn.utils.spectral_norm(nn.Conv2d(CH, c1, 4, 2, padding = 1)), nn.LeakyReLU(0.2,inplace=False),
                                nn.utils.spectral_norm(nn.Conv2d(c1, c2, 4, 2, padding = 1)), nn.LeakyReLU(0.2,inplace=False),
                                nn.utils.spectral_norm(nn.Conv2d(c2, c3, 4, 2, padding = 1)), nn.LeakyReLU(0.2,inplace=False),
                                nn.utils.spectral_norm(nn.Conv2d(c3, c4, 4, 2, padding = 1)), nn.LeakyReLU(0.2,inplace=False))
        self.l2 = nn.Conv2d(c4, 1, 4, 1)
        
    def forward(self, x):
        y = self.l1(x)
        y = self.l2(y)
        return  y
    

#Generator Network
class AutoEnc(nn.Module): # 128 -> 29 -> 128
    def __init__(self):
        super(AutoEnc, self).__init__()
        c1,c2,c3,c4 = 1024, 512, 256, 128
        self.E = nn.Sequential(nn.ConvTranspose2d(100, c1, 4, 1), nn.Tanh(),
                               nn.ConvTranspose2d(c1, c2, 4, 2, padding = 1), nn.Tanh(),
                               nn.ConvTranspose2d(c2, c3, 4, 2, padding = 1), nn.Tanh(),
                               nn.ConvTranspose2d(c3, c4, 4, 2, padding = 1), nn.Tanh(),
                               nn.ConvTranspose2d(c4, CH, 4, 2, padding = 1), nn.Tanh())        
    
    def forward(self, x):
        y = self.E(x)
        return  denormalize_imgs(y)


#Gradient Penalty -> Paper: https://arxiv.org/abs/1704.00028
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
    
    cat = dst.cat_dataset()
    cat = torch.from_numpy(cat).to(dtype = torch.float)
      
    #Generator 
    AutoE = AutoEnc().cuda()
    
    #Critic
    E2 = Encoder2().train().cuda()     
    
    optimizerG = torch.optim.Adam(AutoE.parameters(),lr,betas=(beta1, beta2))
    optimizerC = torch.optim.Adam(E2.parameters(),lr,betas=(beta1, beta2))
    
    
    for eps in range(epochs):
        
        idx_ = torch.randperm(cat.shape[0])
        ctr = 1
        
        for b in range(0,cat.shape[0]-batch,batch):            
            
            ###Critic###            
            optimizerC.zero_grad() 
            real = cat[idx_[b:b+batch]].cuda() #not normalized
            
            x = (torch.rand((1,100,1,1)).cuda() * 2) - 1
            fake = AutoE(x)
            real = torch.cat((real,fake),dim=0) # Adding noise to real data to improve generator stability
            
            x = (torch.rand((real.shape[0],100,1,1)).cuda() * 2) - 1
            fake = AutoE(x) #not normalized
                            
            #Real
            y = E2(normalize_imgs(real))
            errC_real = y.mean()
                
            #fake
            y = E2(normalize_imgs(fake.detach()))
            errC_fake = y.mean()
                
            #total err/loss
            penalty = grad_penalty(E2, real, fake, device='cuda') 
            errC = (errC_fake - errC_real) + (Lambda_penalty * penalty)                   
            errC.backward(retain_graph = True)
            optimizerC.step()
            
            
            if ctr % critic_iter == 0:
                ###Generator###
                x = (torch.rand((real.shape[0],100,1,1)).cuda() * 2) - 1
                fake = AutoE(normalize_imgs(x))
                
                optimizerG.zero_grad()            
                y = E2(normalize_imgs(fake))
                errG = -y.mean()
                errG.backward()            
                optimizerG.step()

                
                #Misc.
                Closses.append(errC.item())
                Glosses.append(errG.item())
                
                if ctr%50 == 0:
                
                    print('[%d/%d][%d/%d]\tLoss_C: %.4f\tLoss_G: %.4f'% (eps, epochs, int(b/batch), int(len(idx_)/batch), errC.item() - penalty, errG.item()))  
                    dst.visualize(fake[0].cpu().detach().numpy().astype(np.uint8))  
                    dst.visualize_25(fake[:25].cpu().detach().numpy().astype(np.uint8))
                
            ctr += 1  
      
    return AutoE, E2, Closses, Glosses


AE,E2,Dl,Gl = advers_train(lr=1E-4,epochs=EPOCHS,batch=64,critic_iter=5)

dst.plt.figure(figsize=(10,5))
dst.plt.title("Generator and Discriminator Loss During Training")
dst.plt.plot(Dl, label='D_loss')
dst.plt.plot(Gl, label='G_loss')
dst.plt.legend()
dst.plt.xlabel("iterations")
dst.plt.ylabel("Loss")
dst.plt.legend()
dst.plt.show()

torch.save(AE.state_dict(), r"temp-gen.pth")
torch.save(E2.state_dict(), r"temp-crit.pth")

noise = torch.rand((1,100,1,1)).cuda()
warped = AE(noise)
wd = warped.cpu().detach().numpy().astype(np.uint8)
print(wd[0,:,32])
dst.visualize(wd[0])
