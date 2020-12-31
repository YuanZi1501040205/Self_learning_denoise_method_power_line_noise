"""models_zoo_alpha.py: File to stores Neural Network architectures for Self_learning_denoise_method_power_line_noise"""

__author__ = "Yuan Zi"
__email__ = "yzi2@central.uh.edu"
__version__ = "1.0.0"


def models(name_model, learn_ratio):
    """ This class defines neural networks modeling
        Set the necessary parameters for different models
        input:
            name_model is a string that specifies the neural network modeling approach
        Return:
            model: pytorch
            loss_function
            optimizer
    """
    import torch
    import torch.nn as nn

    # Time Domain
    if name_model == "HashResUNet1":  #
        #  model build
        # %%
        from Functions import DoubleConv
        from Functions import ResDown
        from Functions import Up
        from Functions import OutConv


        class UNet(nn.Module):
            def __init__(self, input_channels,  bilinear=True):
                super(UNet, self).__init__()
                self.input_channels = input_channels
                self.bilinear = bilinear

                # self.in_attention = SALayer()
                # encoder
                self.inc = DoubleConv(input_channels, 64)
                self.down1 = ResDown(64, 128)
                self.down2 = ResDown(128, 256)
                self.down3 = ResDown(256, 512)
                self.down4 = ResDown(512, 1024)
                self.down5 = ResDown(1024, 2048)
                factor = 2 if bilinear else 1
                self.down6 = ResDown(2048, 4096 // factor)
                # decoder for signal
                self.up1 = Up(4096, 2048 // factor, bilinear)
                self.up2 = Up(2048, 1024 // factor, bilinear)
                self.up3 = Up(1024, 512 // factor, bilinear)
                self.up4 = Up(512, 256, bilinear)
                self.up5 = Up(256, 128, bilinear)
                self.up6 = Up(128, 64, bilinear)
                self.out_sig = OutConv(64, 1)
                # decoder for noise
                self.noise_up1 = Up(4096, 2048 // factor, bilinear)
                self.noise_up2 = Up(2048, 1024 // factor, bilinear)
                self.noise_up3 = Up(1024, 512 // factor, bilinear)
                self.noise_up4 = Up(512, 256, bilinear)
                self.noise_up5 = Up(256, 128, bilinear)
                self.noise_up6 = Up(128, 64, bilinear)
                self.out_noise = OutConv(64, 1)


            def forward(self, x):
                # atten_x = self.in_attention(x)
                x1 = self.inc(x)
                x2 = self.down1(x1)
                x3 = self.down2(x2)
                x4 = self.down3(x3)
                x5 = self.down4(x4)
                x6 = self.down5(x5)
                x7 = self.down6(x6)
                sig = self.up1(x7, x6)
                sig = self.up2(sig, x5)
                sig = self.up3(sig, x4)
                sig = self.up4(sig, x3)
                sig = self.up5(sig, x2)
                sig = self.up6(sig, x1)
                sig = self.out_sig(sig)
                sig = sig.view(sig.shape[0], -1)
                noise = self.noise_up1(x7, x6)
                noise = self.noise_up2(noise, x5)
                noise = self.noise_up3(noise, x4)
                noise = self.noise_up4(noise, x3)
                noise = self.noise_up5(noise, x2)
                noise = self.noise_up6(noise, x1)
                noise = self.out_noise(noise)
                noise = sig.view(noise.shape[0], -1)
                return sig, noise

        model = UNet(input_channels=1, bilinear=True)
        # case we will use self define Mean Squared Error (MSE) as  ur loss function.
        from Functions import my_powerline_loss1
        loss_func = my_powerline_loss1
        optimizer = torch.optim.Adam(model.parameters(), lr=learn_ratio)

    # %%
    elif name_model == "CNN20_AttenResUNet":  # !!! THis is for Time Domain Model
        #  model build
        import torch.nn.functional as F
        from Functions import DoubleConv
        from Functions import Down
        from Functions import ResDown
        from Functions import Up
        from Functions import OutConv
        from Functions import SpatialAttention
        from Functions import AttenResDown


        class UNet(nn.Module):
            def __init__(self, input_channels, bilinear=True):
                super(UNet, self).__init__()
                self.input_channels = input_channels
                self.bilinear = bilinear

                # self.in_attention = SALayer()
                self.inc = DoubleConv(input_channels, 64)
                self.down1 = AttenResDown(64, 128)
                self.down2 = ResDown(128, 256)
                self.down3 = ResDown(256, 512)
                self.down4 = ResDown(512, 1024)
                self.down5 = ResDown(1024, 2048)
                factor = 2 if bilinear else 1
                self.down6 = ResDown(2048, 4096 // factor)
                self.up1 = Up(4096, 2048 // factor, bilinear)
                self.up2 = Up(2048, 1024 // factor, bilinear)
                self.up3 = Up(1024, 512 // factor, bilinear)
                self.up4 = Up(512, 256, bilinear)
                self.up5 = Up(256, 128, bilinear)
                self.up6 = Up(128, 64, bilinear)
                self.outc = OutConv(64, 1)

            def forward(self, x):
                # atten_x = self.in_attention(x)
                x1 = self.inc(x)
                x2 = self.down1(x1)
                x3 = self.down2(x2)
                x4 = self.down3(x3)
                x5 = self.down4(x4)
                x6 = self.down5(x5)
                x7 = self.down6(x6)
                x = self.up1(x7, x6)
                x = self.up2(x, x5)
                x = self.up3(x, x4)
                x = self.up4(x, x3)
                x = self.up5(x, x2)
                x = self.up6(x, x1)
                x = self.outc(x)
                x = x.view(x.shape[0], -1)
                return x

        model = UNet(input_channels=1, bilinear=True)
        # case we will use self define Mean Squared Error (MSE) as  ur loss function.
        loss_func = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), weight_decay=0.0001, lr=0.01, momentum=0.9)

    return model, loss_func, optimizer
