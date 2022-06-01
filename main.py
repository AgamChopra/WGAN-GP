import torch
import sys 
sys.path.append(r"E:\ML\Dog-Cat-GANs")
import wgpmodels as models
import gans1dset as dst

def gen_img(N = 10):
    with torch.no_grad():        
        noise = torch.rand((N,1,10,10))
        
        Gsave = r"E:\ML\Dog-Cat-GANs\Gen_temp.pt"
        Csave = r"E:\ML\Dog-Cat-GANs\Crit_temp.pt"
        
        Gen = models.Generator('A',3).eval()
        Crit = models.Critic('B',3).eval()
        
        try:
            Gen.load_state_dict(torch.load(Gsave))
            Crit.load_state_dict(torch.load(Csave))
        except:
            print('Warning: Could not load generator parameters at',Gsave)
            
    warped = Gen(noise)
    print(warped.shape)
    scores = Crit(warped)
    print(scores.shape)
    
    for i in range(N):
        wd = warped[i].cpu().detach().numpy()
        dst.visualize(wd,dark=False,title='Fake Critic Score = %d'%(scores[i])) 
        
    x = dst.torch_celeb_dataset_sample(N)/255
    scores = Crit(x)
    
    for i in range(x.shape[0]):
        x_ = x[i].cpu().detach().numpy()
        dst.visualize(x_,dark=False,title='Real Critic Score = %d'%(scores[i]))
    
def main():
    gen_img(20)


if __name__ == "__main__":
    main()