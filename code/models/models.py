import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):  # (N,3,64,64) -> (N,1)
    def __init__(self, input_channels, num_feature_maps=256):
        super(Critic, self).__init__()
        self.c = num_feature_maps

        self.feature_extractor = nn.Sequential(
            MultiKernelConv2d(input_channels, self.c, stride=2,
                              padding=1, apply_spectral_norm=True,
                              groups=1),
            nn.Mish(),
            MultiKernelConv2d(self.c, self.c, stride=2,
                              padding=1, apply_spectral_norm=True,
                              groups=2),
            nn.Mish(),
            MultiKernelConv2d(self.c, self.c * 2, stride=2,
                              padding=1, apply_spectral_norm=True,
                              groups=4),
            nn.Mish(),
            MultiKernelConv2d(self.c * 2, self.c * 4, stride=2,
                              padding=1, apply_spectral_norm=True,
                              groups=4),
            nn.Mish(),
            MultiKernelConv2d(self.c * 4, self.c * 8, stride=2,
                              padding=1, apply_spectral_norm=True,
                              groups=1),
            nn.Mish(),
            nn.AdaptiveAvgPool2d(1)
        )

        self.fully_connected = nn.Sequential(nn.Linear(self.c * 8, 1))

    def forward(self, input_tensor):
        features = self.feature_extractor(input_tensor)
        features = self.fully_connected(features.squeeze())
        return features


class Generator(nn.Module):  # (N,noise_dim,1,1) -> (N,3,64,64)
    def __init__(self, input_channels, noise_dim=100, initial_spatial_size=4,
                 num_initial_feature_maps=256, num_feature_maps=256):
        super(Generator, self).__init__()
        self.a = initial_spatial_size
        self.c_ = num_initial_feature_maps
        self.c = num_feature_maps

        self.fully_connected = nn.Sequential(
            nn.Linear(noise_dim, self.c_ * self.a * self.a),
            nn.Mish(),
            nn.BatchNorm1d(self.c_ * self.a * self.a)
        )

        self.feature_extractor = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            MultiKernelConv2d(self.c_, self.c),
            nn.Mish(),
            nn.BatchNorm2d(self.c),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            MultiKernelConv2d(self.c, self.c),
            nn.Mish(),
            nn.BatchNorm2d(self.c),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            MultiKernelConv2d(self.c, self.c),
            nn.Mish(),
            nn.BatchNorm2d(self.c),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            MultiKernelConv2d(self.c, self.c),
            nn.Mish(),
            nn.BatchNorm2d(self.c),
            nn.Conv2d(self.c, input_channels, kernel_size=1),
            nn.Tanh()
        )

    def forward(self, input_tensor):
        features = self.fully_connected(
            input_tensor.view(input_tensor.shape[0], 100))
        features = self.feature_extractor(features.view(
            input_tensor.shape[0], self.c_, self.a, self.a))
        output = (features + 1) / 2
        return output


class MultiKernelConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[1, 3, 5, 7],
                 stride=1, padding=0, groups=1, apply_spectral_norm=False):
        super(MultiKernelConv2d, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels // len(kernel_sizes),
                      kernel_size=k, stride=stride, padding=k // 2,
                      groups=groups)
            for k in kernel_sizes
        ])
        if apply_spectral_norm:
            self.convs = nn.ModuleList(
                [nn.utils.spectral_norm(conv) for conv in self.convs])
        self.padding = padding

    def forward(self, input_tensor):
        out = torch.cat([conv(input_tensor) for conv in self.convs], dim=1)
        if self.padding > 0:
            out = F.pad(out, (self.padding, self.padding,
                        self.padding, self.padding))
        return out


def main():
    print('cuda detected:', torch.cuda.is_available())
    torch.set_printoptions(precision=9)

    # 'highest', 'high', 'medium'. 'highest' is slower but accurate while 'medium'
    #  is faster but less accurate. 'high' is preferred setting. Refer:
    #  https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
    torch.set_float32_matmul_precision('high')

    # 'True' = faster but less accurate, 'False' = Slower but more accurate
    #  has to be set to True if presision is high or medium
    torch.backends.cuda.matmul.allow_tf32 = True

    # 'True' = faster but less accurate, 'False' = Slower but more accurate
    #  has to be set to True if presision is high or medium
    torch.backends.cudnn.allow_tf32 = True

    # For stability 'False', 'True', might be slower
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    # Define input tensors
    input_critic = torch.randn(64, 3, 64, 64).cuda()
    input_generator = torch.randn(64, 100, 1, 1).cuda()

    # Initialize models
    critic = Critic(input_channels=3).cuda()
    generator = Generator(input_channels=3).cuda()

    # Test Generator forward pass
    generator_output = generator(input_generator)
    print("Generator Output shape:", generator_output.shape)

    # Test Critic forward pass
    critic_output = critic(input_critic)
    print("Critic Output:", critic_output.shape)


if __name__ == "__main__":
    main()
