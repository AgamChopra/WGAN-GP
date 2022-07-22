import torch
from PIL import Image
import models
import os


def gen_img(N = 10,Gen_path = r"Gen_temp.pt",Out_path = r"/output",device='cpu'):
    with torch.no_grad():
        print('Output Path: ' + Out_path)

        print('device: %s\nGenerating model ind initial conditions... '%(device))
        Gen = models.GeneratorA(3).eval().to(device)
        print('Model loaded')

        try:
            Gen.load_state_dict(torch.load(Gen_path,map_location=device))
            print('Model parameters loaded')
            Flag = True

        except:
            print('ERROR: Could not load generator parameters at',Gen_path,'\nEXIT')
            raise()
            Flag = False

        if Flag:
            noise = torch.rand((N,1,10,10)).to(device)
            print('Initial conditions generated')

            warped = Gen(noise)
            print('Images generated')

            for i in range(N):
                print('Writing image %d to disk...'%(i))
                wd = (warped[i].detach().cpu() * 255).to(dtype=torch.uint8).numpy().T[:,:,::-1]
                im = Image.fromarray(wd)
                im.save(Out_path + '/output %d.png'%(i))
                print('done')


if __name__ == "__main__":
    gen_img(10,'Gen_temp.pt',os.path.abspath('/home/gans_test/otp'))
