'''
My implementation of a Wasserstein Generative Adversarial 
Network(WGAN) with Gradient Penelty(-GP) for Lipschitz constraint 
and generator added noise to real data for improved Critic stability.
Resources:
    https://arxiv.org/abs/1701.07875
    https://arxiv.org/abs/1704.00028
'''
import torch
from tqdm import trange
import wgpmodels as models
import gans1dset as dst


print('cuda detected:',torch.cuda.is_available())


def grad_penalty(critic, real, fake, device='cpu'):
    b_size, c, h, w = real.shape
    epsilon = torch.rand(b_size, 1, 1, 1).repeat(1,c,h,w).to(device)
    interp_img = real * epsilon + fake * (1 - epsilon)
    
    mix_score = critic(interp_img)
    
    grad = torch.autograd.grad(outputs = mix_score, 
                                   inputs = interp_img, 
                                   grad_outputs=torch.ones_like(mix_score),
                                   create_graph=True,
                                   retain_graph=True)[0]
    
    grad = grad.view(grad.shape[0],-1)
    grad_norm = grad.norm(2,dim=1)
    penalty = torch.mean((grad_norm - 1) ** 2)
    return penalty


def advers_train(dataset, lr = 1E-4, epochs = 5, batch=32, beta1=0.5, beta2=0.999, critic_iter=5, Lambda_penalty = 10, dmt = 16, load_state = False,gmod='A',cmod='A',state=None):
    
    if batch >= dmt:
        REG = int(batch / dmt)
        batch -=  REG    
    else:
        REG = 1
    
    Closses = []
    Glosses = []
    
    CH = dataset.shape[1]
    
    print('loading generator...', end =" ")
    #Generator 
    Gen = models.Generator(gmod,CH).cuda()
    print('done.')
    
    print('loading critic...', end =" ")
    #Critic
    Crit = models.Critic(cmod,CH).cuda()   
    print('done.')
    
    if load_state:
        print('loading previous run state...', end =" ")
        Gen.load_state_dict(torch.load(r"E:\ML\Dog-Cat-GANs\Gen-Autosave.pt"))
        Crit.load_state_dict(torch.load(r"E:\ML\Dog-Cat-GANs\Crit-Autosave.pt"))  
        print('done.')
        
    if state is not None:
        Gen.load_state_dict(torch.load(state[0]))
        Crit.load_state_dict(torch.load(state[1]))  
    
    optimizerG = torch.optim.Adam(Gen.parameters(),lr,betas=(beta1, beta2))
    optimizerC = torch.optim.Adam(Crit.parameters(),lr,betas=(beta1, beta2))
    
    checkpoint = 0
    checkpoint_log = [['Checkpoint','Epoche','Ctr_batch','Critic_Error','Generator_Error','Critic_Penalty']]
    
    print('optimizing...')
    for eps in trange(epochs):
        
        idx_ = torch.randperm(dataset.shape[0])
        ctr = 1
        
        for b in range(0,dataset.shape[0]-batch,batch):            
            
            ###Critic###            
            optimizerC.zero_grad() 
            real = dataset[idx_[b:b+batch]].cuda() / 255.
            
            x = torch.rand((REG,1,10,10)).cuda()
            fake = Gen(x)
            real = torch.cat((real,fake),dim=0) # Adding noise to real data to improve generator stability and weaken critic
            
            x = torch.rand((real.shape[0],1,10,10)).cuda()
            fake = Gen(x) 
                            
            #Real
            y = Crit(real)
            errC_real = y.mean()
                
            #fake
            y = Crit(fake.detach())
            errC_fake = y.mean()
                
            #total err/loss
            penalty = Lambda_penalty * grad_penalty(Crit, real, fake, device='cuda') 
            errC = errC_fake - errC_real + penalty                   
            errC.backward(retain_graph = True)
            optimizerC.step()         
            
            if ctr % critic_iter == 0:
                ###Generator###
                x = torch.rand((real.shape[0],1,10,10)).cuda()
                fake = Gen(x)
                
                optimizerG.zero_grad()            
                y = Crit(fake)
                errG = -y.mean()
                errG.backward()            
                optimizerG.step()
                
                #Misc.
                Closses.append(errC.item())
                Glosses.append(errG.item())
                
                if ctr % (critic_iter * 100) == 0:
                    print('[%d/%d][%d/%d]\tLoss_C: %.4f\tLoss_G: %.4f'% (eps, epochs, int(b/batch), int(len(idx_)/batch), errC.item(), errG.item()))  
                    dst.visualize_25(fake[:25].cpu().detach().numpy(),dark = False) #dst.visualize(fake[0].cpu().detach().numpy())
                    torch.save(Gen.state_dict(), r"E:\ML\Dog-Cat-GANs\Gen-Autosave.pt")
                    torch.save(Crit.state_dict(), r"E:\ML\Dog-Cat-GANs\Crit-Autosave.pt")
               
                if ctr % (critic_iter * 600) == 0:
                    print('Saving checkpoint #%d @ epoch %d, ctr_batch %d, LossC,LossG = [%d, %d], C_penalty = %d'%(checkpoint,eps,ctr,errC.item(),errG.item(),penalty))
                    checkpoint_log.append([checkpoint,eps,ctr,errC.item(),errG.item(),penalty])
                    torch.save(Gen.state_dict(), r"E:\ML\Dog-Cat-GANs\Gen_checkpoint_%d.pt"%(checkpoint))
                    torch.save(Crit.state_dict(), r"E:\ML\Dog-Cat-GANs\Crit_checkpoint_%d.pt"%(checkpoint))
                    checkpoint += 1
                    
            ctr += 1
                 
      
    return Gen, Crit, Closses, Glosses, checkpoint_log


def train(gmod = 'A',cmod = 'B', Gsave = r"E:\ML\Dog-Cat-GANs\Gen_temp.pt", Csave = r"E:\ML\Dog-Cat-GANs\Crit_temp.pt"):
    
    print('loading data...')
    dataset = dst.torch_celeb_dataset()
    print('done.')

    EPOCHS = 80
    print("E:\ML\Dog-Cat-GANs\Gen-Autosave.pt")
    print("E:\ML\Dog-Cat-GANs\Crit-Autosave.pt")
    Gen,Crit,Dl,Gl = advers_train(dataset=dataset,lr=1E-4,epochs=EPOCHS,batch=64,critic_iter=5,load_state = False,gmod=gmod,cmod=cmod)#,state=[Gsave,Csave])
    
    dst.plt.figure(figsize=(10,5))
    dst.plt.title("Generator and Critic Loss During Training")
    dst.plt.plot(Dl, label='D_loss')
    dst.plt.plot(Gl, label='G_loss')
    dst.plt.legend()
    dst.plt.xlabel("iterations")
    dst.plt.ylabel("Loss")
    dst.plt.legend()
    dst.plt.show()
    
    torch.save(Gen.state_dict(), Gsave)
    torch.save(Crit.state_dict(), Csave)

def gen_img(gmod = 'A'):
    Gsave = r"E:\ML\Dog-Cat-GANs\Gen-Autosave.pt"
    Gen = models.Generator(gmod,3).cuda()
    try:
        Gen.load_state_dict(torch.load(Gsave))
    except:
        print('Warning: Could not load generator parameters at',Gsave)
    noise = torch.rand((25,1,10,10)).cuda()
    warped = Gen(noise)
    wd = warped.cpu().detach().numpy()
    print(warped.shape)
    dst.visualize_25(wd,dark=False)
    
def main():
    x = int(input('Would you like to train model(press \'1\') or generate synthetic images from previous state(press \'2\')?'))
    if x == 1:
        train()
    elif x==2:
        gen_img()
    else:
        print('Value Error: Please enter either 1 or 2')

if __name__ == "__main__":
    main()