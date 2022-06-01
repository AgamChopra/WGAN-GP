import torch.nn as nn
import sys 
sys.path.append(r"E:\ML\Dog-Cat-GANs")
import myViT as vit


#Critic    
class CriticA(nn.Module): # (N,3,128,128) -> (N,1)
    def __init__(self,CH):
        super(CriticA, self).__init__()
        c1,c2,c3,c4,c5 = 64, 128, 256, 512, 1024
        self.l1 = nn.Sequential(nn.utils.spectral_norm(nn.Conv2d(CH, c1, 4, 2, padding = 1)), nn.LeakyReLU(0.2,inplace=False),
                                nn.utils.spectral_norm(nn.Conv2d(c1, c2, 4, 2, padding = 1)), nn.LeakyReLU(0.2,inplace=False),
                                nn.utils.spectral_norm(nn.Conv2d(c2, c3, 4, 2, padding = 1)), nn.LeakyReLU(0.2,inplace=False),
                                nn.utils.spectral_norm(nn.Conv2d(c3, c4, 4, 2, padding = 1)), nn.LeakyReLU(0.2,inplace=False),
                                nn.utils.spectral_norm(nn.Conv2d(c4, c5, 4, 2, padding = 1)), nn.LeakyReLU(0.2,inplace=False))
        self.l2 = nn.Conv2d(c5, 1, 4, 1)
        
    def forward(self, x):
        y = self.l1(x)
        y = self.l2(y).squeeze()
        return  y   
    
    
class CriticB(nn.Module): # (N,3,128,128) -> (N,1)
    def __init__(self,CH):
        super(CriticB, self).__init__()
        self.c = 40
        
        self.E = nn.Sequential(nn.utils.spectral_norm(nn.Conv2d(CH, self.c, kernel_size=3, stride=1)), nn.LeakyReLU(0.2,inplace=False),
                               nn.utils.spectral_norm(nn.Conv2d(self.c, self.c, kernel_size=3, stride=2)), nn.LeakyReLU(0.2,inplace=False),
                               nn.utils.spectral_norm(nn.Conv2d(self.c, self.c, kernel_size=3, stride=1)), nn.LeakyReLU(0.2,inplace=False),
                               nn.utils.spectral_norm(nn.Conv2d(self.c, self.c, kernel_size=3, stride=2)), nn.LeakyReLU(0.2,inplace=False),
                               nn.utils.spectral_norm(nn.Conv2d(self.c, self.c, kernel_size=3, stride=1)), nn.LeakyReLU(0.2,inplace=False),
                               nn.utils.spectral_norm(nn.Conv2d(self.c, self.c, kernel_size=3, stride=2)), nn.LeakyReLU(0.2,inplace=False),
                               nn.utils.spectral_norm(nn.Conv2d(self.c, self.c, kernel_size=3, stride=1)), nn.LeakyReLU(0.2,inplace=False),
                               nn.utils.spectral_norm(nn.Conv2d(self.c, self.c, kernel_size=3, stride=1)), nn.LeakyReLU(0.2,inplace=False),
                               nn.utils.spectral_norm(nn.Conv2d(self.c, self.c, kernel_size=3, stride=2)), nn.LeakyReLU(0.2,inplace=False),
                               nn.MaxPool2d(kernel_size=4, padding=0, stride=4))     
        
        self.fc = nn.Sequential(nn.Linear(self.c, 1))
        
    def forward(self, x):
        y = self.E(x)
        y = self.fc(y.squeeze())
        return  y    
    
    
class CriticC(nn.Module): # (N,3,128,128) -> (N,1)
    def __init__(self,CH):
        super(CriticC, self).__init__()
        self.l1 = nn.Sequential(nn.utils.spectral_norm(nn.Conv2d(CH, 64, 4, 2, padding = 1)), nn.LeakyReLU(0.2,inplace=False))
        self.l2 = vit.VisionTransformer(img_size = 64, patch_size = 16, in_c = 64, n_classes = 1, depth = 5)
        
    def forward(self, x):
        y = self.l1(x)
        y = self.l2(y).squeeze()
        return  y 
         
    
def Critic(mode='A',CH=3):
    if mode == 'A':
        model = CriticA(CH)
    elif mode == 'B':
        model = CriticB(CH)
    elif mode == 'C':
        model = CriticC(CH)
    else:
        model = None
    return model    


#Generator. Returns floating point RGB (3 channel with continous range of [0.,255.]) images.
class GeneratorB(nn.Module): # 128 -> 29 -> 128
    def __init__(self,CH):
        super(GeneratorB, self).__init__()
        c1,c2,c3,c4,c5 = 1024, 512, 256, 128, 64
        self.E = nn.Sequential(nn.ConvTranspose2d(100, c1, 4, 1), nn.Tanh(),
                               nn.ConvTranspose2d(c1, c2, 4, 2, padding = 1), nn.Tanh(),
                               nn.ConvTranspose2d(c2, c3, 4, 2, padding = 1), nn.Tanh(),
                               nn.ConvTranspose2d(c3, c4, 4, 2, padding = 1), nn.Tanh(),
                               nn.ConvTranspose2d(c4, c5, 4, 2, padding = 1), nn.Tanh(),
                               nn.ConvTranspose2d(c5, CH, 4, 2, padding = 1), nn.Tanh())        
    
    def forward(self, x):
        y = self.E(x)
        output = (y + 1)/2 #((y - y.min())/(y.max() - y.min()))
        return  output
    
    
class GeneratorA(nn.Module): # (N,100,1,1) -> (N,3,128,128)
    def __init__(self,CH):
        super(GeneratorA, self).__init__()
        self.a,self.c_,self.c = 16,24,40
        
        self.fc = nn.Sequential(nn.Linear(100, self.c_ * self.a * self.a), nn.ReLU(inplace=False), nn.BatchNorm1d(self.c_ * self.a * self.a))
        
        self.E = nn.Sequential(nn.ConvTranspose2d(self.c_, self.c, kernel_size=4, padding=1, stride=2), nn.ReLU(inplace=False), nn.BatchNorm2d(self.c),
                               nn.Conv2d(self.c, self.c, kernel_size=3, padding=1, stride=1), nn.ReLU(inplace=False), nn.BatchNorm2d(self.c),
                               nn.Conv2d(self.c, self.c, kernel_size=3, padding=1, stride=1), nn.ReLU(inplace=False), nn.BatchNorm2d(self.c),
                               nn.Conv2d(self.c, self.c, kernel_size=3, padding=1, stride=1), nn.ReLU(inplace=False), nn.BatchNorm2d(self.c),
                               nn.ConvTranspose2d(self.c, self.c, kernel_size=4, padding=1, stride=2), nn.ReLU(inplace=False), nn.BatchNorm2d(self.c),
                               nn.Conv2d(self.c, self.c, kernel_size=3, padding=1, stride=1), nn.ReLU(inplace=False), nn.BatchNorm2d(self.c),
                               nn.ConvTranspose2d(self.c, self.c, kernel_size=4, padding=1, stride=2), nn.ReLU(inplace=False), nn.BatchNorm2d(self.c),
                               nn.Conv2d(self.c, CH, kernel_size=3, padding=1, stride=1), nn.Tanh())        
    
    def forward(self, x):
        y = self.fc(x.view(x.shape[0],100))
        y = self.E(y.view(x.shape[0],self.c_,self.a,self.a))
        output = (y + 1)/2
        return  output
    
    
class GeneratorC(nn.Module):
    def __init__(self,CH):
        super(GeneratorC, self).__init__()
        c1,c2,c3 = 512, 128, 32
        self.F = nn.Sequential(nn.ConvTranspose2d(1, c1, 4, 3), nn.Tanh(),
                                nn.ConvTranspose2d(c1, c2, 3, 2), nn.Tanh(),
                                nn.ConvTranspose2d(c2, c3, 3, 2), nn.Tanh(),
                                nn.ConvTranspose2d(c3, CH, 2, 1), nn.Sigmoid())

    def forward(self, x):
        output = self.F(x)
        return  output
    
    
def Generator(mode='A',CH=3):
    if mode == 'A':
        model = GeneratorA(CH)
    elif mode == 'B':
        model = GeneratorB(CH)
    elif mode == 'C':
        model = GeneratorC(CH)
    else:
        model = None
    return model


# =============================================================================
# import torch  
# torch.manual_seed(0)
# x = torch.ones(64,3,128,128).cuda()
# m = CriticB(3).cuda()
# y = m(x)
# print(y.shape,y.mean()) 
# =============================================================================
# =============================================================================
# import torch  
# torch.manual_seed(0)
# x = torch.rand((64,1,10,10)).cuda()
# m = Generator('A').cuda()
# y = m(x)
# print(y.shape)
# =============================================================================
