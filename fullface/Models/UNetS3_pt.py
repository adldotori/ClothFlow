from Models.networks_general import *
from Models.PartialConv_2d import *


def conv3x3(in_channels, out_channels, stride=1,
            padding=1, bias=True, groups=1):
    return PartialConv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)


def upconv2x2(in_channels, out_channels, mode='transpose'):
    if mode == 'transpose':
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2)
    else:
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            conv3x3(in_channels, out_channels))
        return nn.Sequential(
            nn.Upsample(scale_factor=2),
            conv3x3(in_channels, out_channels))


def conv1x1(in_channels, out_channels, groups=1):
    return PartialConv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1)


class DownConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """

    def __init__(self, in_channels, out_channels, pooling=True, norm_layer=nn.InstanceNorm2d):
        super(DownConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling
        use_bias = norm_layer == nn.InstanceNorm2d
        self.conv1 = nn.Sequential(PartialConv2d(in_channels, out_channels, 3, 1, 1, bias=use_bias),
                                   norm_layer(out_channels),
                                   nn.LeakyReLU(0.2, True))
        self.conv2 = nn.Sequential(PartialConv2d(out_channels, out_channels, 3, 1, 1, bias=use_bias),
                                   norm_layer(out_channels),
                                   nn.LeakyReLU(0.2, True))

        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        before_pool = x
        if self.pooling:
            x = self.pool(x)

        return x, before_pool


class UpConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU activation follows each convolution.
    """

    def __init__(self, in_channels, out_channels,
                 merge_mode='concat', up_mode='transpose'):
        super(UpConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode

        self.upconv = upconv2x2(self.in_channels, self.out_channels,
                                mode=self.up_mode)

        if self.merge_mode == 'concat':
            self.conv1 = nn.Sequential(
                PartialConv2d(2 * self.out_channels, self.out_channels, kernel_size=3, padding=1, stride=1),
                nn.InstanceNorm2d(self.out_channels),
                nn.LeakyReLU(0.2, True)
            )
        else:
            # num of input channels to conv2 is same
            self.conv1 = conv3x3(self.out_channels, self.out_channels)
        self.conv2 = nn.Sequential(
                PartialConv2d(self.out_channels, self.out_channels,3, 1,1),
                nn.InstanceNorm2d(self.out_channels),
                nn.LeakyReLU(0.2, True)
            )

    def forward(self, from_A, from_up):
        """ Forward pass
        Arguments:
            from_down: tensor from the encoder pathway
            from_up: upconv'd tensor from the decoder pathway
        """
        # print(from_up.shape)
        from_up = self.upconv(from_up)
        # print(from_up.shape)
        if self.merge_mode == 'concat':
            x = torch.cat((from_up, from_A), 1)
        else:
            x = from_up + from_A

        x = self.conv1(x)

        x = self.conv2(x)
        return x


class UNet(nn.Module):

    def __init__(self, opt, in_channels=19, depth=5,
                 start_filts=64, up_mode='transpose',
                 merge_mode='concat'):
        """
        Arguments:
            in_channels: int, number of channels in the input tensor.
                Default is 3 for RGB images.
            depth: int, number of MaxPools in the U-Net.
            start_filts: int, number of convolutional filters for the
                first conv.
            up_mode: string, type of upconvolution. Choices: 'transpose'
                for transpose convolution or 'upsample' for nearest neighbour
                upsampling.
        """
        super(UNet, self).__init__()

        if up_mode in ('transpose', 'upsample'):
            self.up_mode = up_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for "
                             "upsampling. Only \"transpose\" and "
                             "\"upsample\" are allowed.".format(up_mode))

        if merge_mode in ('concat', 'add'):
            self.merge_mode = merge_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for"
                             "merging up and down paths. "
                             "Only \"concat\" and "
                             "\"add\" are allowed.".format(up_mode))

        # NOTE: up_mode 'upsample' is incompatible with merge_mode 'add'
        if self.up_mode == 'upsample' and self.merge_mode == 'add':
            raise ValueError("up_mode \"upsample\" is incompatible "
                             "with merge_mode \"add\" at the moment "
                             "because it doesn't make sense to use "
                             "nearest neighbour to reduce "
                             "depth channels (by half).")


        self.in_channels = in_channels
        self.start_filts = start_filts
        self.depth = depth

        self.down_convs = []
        self.up_convs = []


        """
        self.feature = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.LeakyReLU(0.2, True)
        )
        #person-warp
        
        self.cloth_warp = []
        for i in range(depth):
            ins = 3 if i == 0 else out
            out = self.start_filts * (2 ** i)
            pooling = True if i < depth - 1 else False
            down_conv = DownConv(ins, out, pooling=pooling)
            self.cloth_warp.append(down_conv)
        """

        self.bottle_0 = nn.Sequential(PartialConv2d(start_filts * 2**(depth-1),start_filts * 2**(depth-1),3,1,1),
                                    nn.InstanceNorm2d(start_filts * 2**(depth-1)),
                                    nn.LeakyReLU(0.2, True))
        self.bottle_1 = nn.Sequential(PartialConv2d(start_filts * 2**(depth-1),start_filts * 2**(depth-1),3, dilation=2, padding=2, bias=False),
                                    nn.InstanceNorm2d(start_filts * 2**(depth-1)),
                                    nn.LeakyReLU(0.2, True)
                                    )
        self.bottle_2 = nn.Sequential(PartialConv2d(start_filts * 2**(depth-1), start_filts * 2**(depth-1), 3, dilation=4, padding=4, bias=False),
                                      nn.InstanceNorm2d(start_filts * 2**(depth-1)),
                                      nn.LeakyReLU(0.2, True)
                                      )

        for i in range(depth):
            ins = in_channels if i == 0 else outs
            outs = self.start_filts * (2 ** i)
            pooling = True if i < depth - 1 else False

            down_conv = DownConv(ins, outs, pooling=pooling)
            self.down_convs.append(down_conv)

        for i in range(depth - 1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs, up_mode=up_mode,
                             merge_mode=merge_mode)
            self.up_convs.append(up_conv)


        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)
        #self.cloth_warp = nn.ModuleList(self.cloth_warp)
        self.final_layer1 = PartialConv2d(start_filts,3,7,1,3)
        #self.final_layer2 = nn.Conv2d(64,1,7,1,3)
        #self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()
        init_weights(self.final_layer1,'xavier')
        #init_weights(self.final_layer2,'xavier')
        #init_weights(self.feature, 'xavier')
        init_weights(self.bottle_0,'xavier')
        init_weights(self.bottle_1,'xavier')
        init_weights(self.bottle_2,'xavier')


        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, PartialConv2d):
            init.xavier_normal(m.weight)
            # init.constant(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, lack):
        encoder_outs = []
        x = lack
        # encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)
    # def forward(self, off_cloth, target_pose, cn_cloth, warped_cloth, head, pants):
    #     encoder_outs = []
    #     x = torch.cat((off_cloth, target_pose, cn_cloth, warped_cloth, head, pants), 1)
    #     # encoder pathway, save outputs for merging
    #     for i, module in enumerate(self.down_convs):
    #         x, before_pool = module(x)
    #         encoder_outs.append(before_pool)

        """
        y = cloth
        encoder_out_cloth = []
        for i, module in enumerate(self.cloth_warp):
            y, before_pool_feature = module(y)
            encoder_out_cloth.append(before_pool_feature)
        """


        x = self.bottle_0(x)
        bottle_0 = x
        x = self.bottle_1(x)
        bottle_1 = x
        x = self.bottle_2(x)
        bottle_2 = x

        x = bottle_0 + bottle_1 + bottle_2


        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i + 2)]
            x = module(before_pool, x)

        rough = self.final_layer1(x)
        rough = self.tanh(rough)

        return rough


