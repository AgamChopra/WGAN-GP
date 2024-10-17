import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import trange
import models
import dataset as dst

print('cuda detected:', torch.cuda.is_available())
torch.set_printoptions(precision=9)

# 'highest', 'high', 'medium'. 'highest' is slower but accurate while 'medium'
#  is faster but less accurate. 'high' is preferred setting. Refer:
#  https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
torch.set_float32_matmul_precision('medium')

# 'True' = faster but less accurate, 'False' = Slower but more accurate
#  has to be set to True if precision is high or medium
torch.backends.cuda.matmul.allow_tf32 = True

# 'True' = faster but less accurate, 'False' = Slower but more accurate
#  has to be set to True if precision is high or medium
torch.backends.cudnn.allow_tf32 = True

# For stability 'False', 'True', might be slower
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


def grad_penalty(critic, real, fake, weight):
    b_size, c, h, w = real.shape
    epsilon = torch.rand(b_size, 1, 1, 1).repeat(1, c, h, w).cuda()
    interp_img = (real * epsilon) + (fake * (1 - epsilon))

    mix_score = critic(interp_img)

    grad = torch.autograd.grad(outputs=mix_score,
                               inputs=interp_img,
                               grad_outputs=torch.ones_like(mix_score).cuda(),
                               create_graph=True,
                               retain_graph=True)[0]

    grad = grad.view(b_size, -1)
    # Calculating manual gradient norm to avoid issues around 0 by adding epsilon = 1E-12. {grad.norm(2,dim=1)}
    grad_norm = torch.sqrt((torch.sum(grad ** 2, dim=1)) + 1E-6)
    penalty = torch.mean((grad_norm - 1) ** 2)
    return weight * penalty


def advers_train(dataloader, lr=1E-4, epochs=5, batch=32, beta1=0.5, beta2=0.999,
                 critic_iter=5, Lambda_penalty=10, dmt=32, load_state=False,
                 state=None):

    REG = int(batch / dmt)
    batch -= REG

    Closses = []
    Glosses = []

    CH = 3

    print('loading generator...', end=" ")
    # Generator
    Gen = models.Generator(CH).cuda()
    print('done.')

    print('loading critic...', end=" ")
    # Critic
    Crit = models.Critic(CH).cuda()
    print('done.')

    if load_state:
        print('loading previous run state...', end=" ")
        Gen.load_state_dict(torch.load(
            "/home/agam/Documents/git_projects/Gen-Autosave.pt"))
        Crit.load_state_dict(torch.load(
            "/home/agam/Documents/git_projects/Crit-Autosave.pt"))
        print('done.')

    if state is not None:
        Gen.load_state_dict(torch.load(state[0]))
        Crit.load_state_dict(torch.load(state[1]))

    optimizerG = torch.optim.Adam(Gen.parameters(), lr, betas=(beta1, beta2))
    optimizerC = torch.optim.Adam(Crit.parameters(), lr, betas=(beta1, beta2))

    print('optimizing...')
    print(dataloader.batch_size, len(dataloader.dataset))

    for eps in trange(epochs):
        ctr = 1

        for i, (real, _) in enumerate(dataloader):
            if i * dataloader.batch_size >= len(dataloader.dataset):
                break

            real = real.cuda()
            optimizerC.zero_grad()

            x = torch.rand((REG, 100, 1, 1)).cuda()
            fake = Gen(x)
            # Adding noise to real data to improve generator stability and weaken critic
            real = torch.cat((real, fake), dim=0)

            x = torch.rand((real.shape[0], 100, 1, 1)).cuda()
            fake = Gen(x)

            # Real
            y = Crit(real)
            errC_real = y.mean()

            # Fake
            y = Crit(fake.detach())
            errC_fake = y.mean()

            # Total err/loss
            penalty = grad_penalty(Crit, real, fake, Lambda_penalty)
            errC = errC_fake - errC_real + penalty
            errC.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(Crit.parameters(), max_norm=1.0)
            optimizerC.step()

            if ctr % critic_iter == 0:
                ### Generator###
                x = torch.rand((real.shape[0], 100, 1, 1)).cuda()
                fake = Gen(x)

                optimizerG.zero_grad()
                y = Crit(fake)
                errG = -y.mean()
                errG.backward()
                torch.nn.utils.clip_grad_norm_(Gen.parameters(), max_norm=1.0)
                optimizerG.step()

                # Misc.
                Closses.append(errC.item())
                Glosses.append(errG.item())

                if ctr % (critic_iter * 100) == 0:
                    print('[%d/%d][%d/%d]\tLoss_C: %.4f\tLoss_G: %.4f' % (eps, epochs,
                          ctr, len(dataloader), errC.item(), errG.item()))
                    dst.visualize_16(
                        fake[:16].cpu().detach().numpy(), dark=True)

            if eps % 10 == 0:
                torch.save(Gen.state_dict(),
                           "/home/agam/Documents/git_projects/Gen-Autosave.pt")
                torch.save(Crit.state_dict(),
                           "/home/agam/Documents/git_projects/Crit-Autosave.pt")

            ctr += 1

    return Gen, Crit, Closses, Glosses


def train():
    EPOCHS = 1000
    BATCH = 128

    print('loading data...')
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    celeb_dataset = datasets.CelebA(
        root='./data', split='train', download=False, transform=transform)
    dataloader = DataLoader(celeb_dataset, batch_size=BATCH, shuffle=True)
    print('done.')

    print("/home/agam/Documents/git_projects/Gen-Autosave.pt")
    print("/home/agam/Documents/git_projects/Crit-Autosave.pt")
    Gen, Crit, Dl, Gl = advers_train(
        dataloader=dataloader, lr=1E-4, epochs=EPOCHS, batch=BATCH,
        dmt=BATCH/2, critic_iter=5, load_state=False
    )

    dst.plt.figure(figsize=(10, 5))
    dst.plt.title("Generator and Critic Loss During Training")
    dst.plt.plot(Dl, label='D_loss')
    dst.plt.plot(Gl, label='G_loss')
    dst.plt.legend()
    dst.plt.xlabel("iterations")
    dst.plt.ylabel("Loss")
    dst.plt.legend()
    dst.plt.show()

    torch.save(Gen.state_dict(),
               "/home/agam/Documents/git_projects/Gen-Autosave.pt")
    torch.save(Crit.state_dict(),
               "/home/agam/Documents/git_projects/Crit-Autosave.pt")

    return Gen


def gen_img(Gen=None):
    if Gen is None:
        Gsave = "/home/agam/Documents/git_projects/Gen_temp.pt"
        Gen = models.Generator(3).cuda()
        try:
            Gen.load_state_dict(torch.load(Gsave))
        except Exception:
            print('Warning: Could not load generator parameters at', Gsave)
    for _ in range(20):
        noise = torch.rand((25, 1, 10, 10)).cuda()
        warped = Gen(noise)
        wd = warped.cpu().detach().numpy()
        print(warped.shape)
        dst.visualize_25(wd, dark=False)


def main():
    x = int(input("Would you like to train model(press '1') or generate synthetic images from previous state(press '2')?"))
    if x == 1:
        train()
    elif x == 2:
        gen_img()
    else:
        print('Value Error: Please enter either 1 or 2')


if __name__ == "__main__":
    main()
