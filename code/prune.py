import torch
from torch import nn
import torch.nn.utils.prune as prune

import models
import dataset as dst


def prune_model_global_unstructured(model, proportion):
    module_tups = []
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.Linear):
            module_tups.append((module, 'weight'))
            
    prune.global_unstructured(parameters=module_tups, pruning_method=prune.L1Unstructured,amount=proportion)
    for module, _ in module_tups:
        prune.remove(module, 'weight')
    return model


def prune_model(N=0.3):
    with torch.no_grad():        
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        noise = torch.rand((10,1,10,10)).to(device)
        
        Gsave = "R:/git projects/WGAN-GP/parameters/Gen_temp.pt"
        Gen = models.Generator('A',3).eval().to(device)
        
        try:
            Gen.load_state_dict(torch.load(Gsave))
        except:
            print('Warning: Could not load generator parameters at',Gsave)
        
        Gen_sparse = prune_model_global_unstructured(model=Gen, proportion=N)

        warped = Gen_sparse(noise)
        print(warped.shape)
        
        for i in range(10):
            wd = warped[i].cpu().detach().numpy()
            dst.visualize(wd,dark=False)
        
        return Gen_sparse
  
    
def main():
    model = prune_model(0.25)
    torch.save(model.state_dict(), "R:/git projects/WGAN-GP/parameters/Gen_pruned.pt")


if __name__ == "__main__":
    main()
