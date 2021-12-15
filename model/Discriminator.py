import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, image_size=512, num_enc_dec_layers=4, num_pool_layers=4, num_in_channels=2):
        super().__init__()

        self.freeze_enc_bn = False
        self.num_enc_dec_layers = num_enc_dec_layers
        self.num_pool_layers = num_pool_layers
        self.num_in_channels = num_in_channels
        self.net_depth = num_enc_dec_layers + num_pool_layers

        # define encoding layers
        net = []
        net.append(
            nn.Conv2d(
                self.num_in_channels,
                image_size // (2 ** (self.num_enc_dec_layers - 1)), (7, 7), (1, 1), (3, 3)))
        net.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        for i in range(1, self.num_enc_dec_layers):
            if i == self.num_enc_dec_layers - 1:
                net.append(
                    nn.Conv2d(
                        image_size // (2 ** (self.num_enc_dec_layers - i)),
                        image_size // (2 ** (self.num_enc_dec_layers - i - 1)),
                        (3, 3), (2, 2), (1, 1)))
            else:
                net.append(nn.Conv2d(
                    image_size // (2 ** (self.num_enc_dec_layers - i)),
                    image_size // (2 ** (self.num_enc_dec_layers - i - 1)),
                    (5, 5), (2, 2), (2, 2)))
            net.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        # define encoding pooling layers
        for i in range(self.num_pool_layers):
            net.append(nn.Conv2d(
                image_size,
                image_size,
                (3, 3), (2, 2), (1, 1)))
            net.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        print(image_size * ((image_size // (2 ** (self.num_enc_dec_layers - 1))) ** 2))
        net.append(nn.Linear(image_size * ((image_size // (2 ** (self.num_enc_dec_layers - 1))) ** 2), image_size))
        net.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        net.append(nn.Linear(image_size, 1))
        net.append(nn.Sigmoid())
        self.conv = nn.Sequential(*net)

    def forward(self, input, mask):
        h = torch.cat([input, mask], dim=1)
        for net in self.conv:
            if isinstance(net, nn.Linear):
                print(h.shape)
                h = torch.flatten(h, start_dim=1)
            h = net(h)
        return h
