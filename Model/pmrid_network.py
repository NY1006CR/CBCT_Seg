import paddle.nn as nn


def DepthwiseConv(in_channels, kernel_size, stride, padding):
    return nn.Conv2D(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride,
                     padding=padding, groups=in_channels, bias_attr=False)


def PointwiseConv(in_channels, out_channels):
    return nn.Conv2D(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0, bias_attr=True)


class CovSepBlock(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=0):
        super().__init__()
        self.dc = DepthwiseConv(in_channels, kernel_size, stride=stride, padding=padding)
        self.pc = PointwiseConv(in_channels, out_channels)
        self.dc2 = DepthwiseConv(out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.dc(x)
        x1 = self.pc(x)
        x = self.dc2(x1) + x1
        return x


def conv(in_channels, out_channels, kernel_size, bias_attr=False, stride=1):
    return nn.Sequential(DepthwiseConv(in_channels, kernel_size, stride=stride, padding=(kernel_size // 2)),
                         PointwiseConv(in_channels, out_channels)
                         )


class CALayer(nn.Layer):
    def __init__(self, channel, reduction=4, bias_attr=False):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.conv_du = nn.Sequential(
            nn.Conv2D(channel, channel // reduction, 1, padding=0, bias_attr=bias_attr),
            nn.ReLU(),
            nn.Conv2D(channel // reduction, channel, 1, padding=0, bias_attr=bias_attr),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class CAB(nn.Layer):
    def __init__(self, n_feat, kernel_size=3, reduction=4, bias_attr=False):
        super(CAB, self).__init__()
        self.CA = CALayer(n_feat, reduction, bias_attr=bias_attr)
        self.body = nn.Sequential(conv(n_feat, n_feat * 2, kernel_size=5, bias_attr=bias_attr),
                                  nn.LeakyReLU(0.125),
                                  conv(n_feat * 2, n_feat, kernel_size=3, bias_attr=bias_attr),
                                  )

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res


class Encoder(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.sepconv = CovSepBlock(in_channels, out_channels * 2, padding=2)
        self.activate = nn.LeakyReLU(0.125)
        self.sepconv2 = CovSepBlock(out_channels * 2, out_channels, padding=2)
        self.proj = nn.Identity()
        if in_channels != out_channels:
            self.proj = CovSepBlock(in_channels, out_channels, kernel_size=3, padding=1)
        self.cab = CAB(out_channels)

    def forward(self, x):
        branch = self.proj(x)
        x = self.sepconv(x)
        x = self.activate(x)
        x = self.sepconv2(x)
        x += branch
        x = self.cab(x)
        return x


class Upsampling(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size=2):
        super().__init__()
        self.upsample = nn.Conv2DTranspose(in_channels, out_channels, kernel_size=kernel_size, stride=2)

    def forward(self, x):
        return self.upsample(x)


class Downsampling(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.sepconv = CovSepBlock(in_channels=in_channels, out_channels=out_channels * 2, stride=2, padding=2)
        self.activate = nn.LeakyReLU(0.125)
        self.sepconv2 = CovSepBlock(in_channels=out_channels * 2, out_channels=out_channels, padding=2)
        self.branchconv = CovSepBlock(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.cab = CAB(out_channels)

    def forward(self, x):
        branch = x
        x = self.sepconv(x)
        x = self.activate(x)
        x = self.sepconv2(x)
        branch = self.branchconv(branch)
        x += branch
        x = self.cab(x)
        return x


class Decoder(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.sepconv = CovSepBlock(in_channels, out_channels, kernel_size=3, padding=1)
        self.activate = nn.LeakyReLU(0.125)
        self.sepconv2 = CovSepBlock(out_channels, out_channels, kernel_size=3, padding=1)
        self.cab = CAB(out_channels)

    def forward(self, x):
        branch = x
        x = self.sepconv(x)
        x = self.activate(x)
        x = self.sepconv2(x)
        x = x + branch
        x = self.cab(x)
        return x


def EncoderStage(in_channels, out_channels, num_encoder):
    seq = [
        Downsampling(in_channels, out_channels),
    ]
    for _ in range(num_encoder):
        seq.append(
            Encoder(out_channels, out_channels)
        )
    return nn.Sequential(*seq)


class DecoderStage(nn.Layer):
    def __init__(self, in_channels, out_channels, skip_in_channels):
        super().__init__()
        self.decoder = Decoder(in_channels, in_channels)
        self.upsampling = Upsampling(in_channels, out_channels)
        self.skipconnect = CovSepBlock(skip_in_channels, out_channels, kernel_size=3, padding=1)
        self.activate = nn.LeakyReLU(0.125)
        self.cab = CAB(out_channels)

    def forward(self, x):
        input, skip = x
        input = self.decoder(input)
        input = self.upsampling(input)
        skip = self.skipconnect(skip)
        skip = self.activate(skip)
        output = input + skip
        output = self.cab(output)
        return output


class Net(nn.Layer):
    def __init__(self):
        super().__init__()

        chans0 = 3
        chans1 = 32
        chans2 = 64
        chans3 = 128
        chans4 = 256

        self.conv = nn.Conv2D(in_channels=chans0, out_channels=chans1, kernel_size=5, padding=2)
        self.relu = nn.LeakyReLU(0.125)
        self.encoder_stage1 = EncoderStage(in_channels=chans1, out_channels=chans2, num_encoder=1)
        self.encoder_stage2 = EncoderStage(in_channels=chans2, out_channels=chans3, num_encoder=1)
        self.encoder_stage3 = EncoderStage(in_channels=chans3, out_channels=chans4, num_encoder=1)

        self.enc2dec = CovSepBlock(in_channels=chans4, out_channels=chans4, kernel_size=3, padding=1)
        self.med_activate = nn.LeakyReLU(0.125)

        self.decoder_stage2 = DecoderStage(in_channels=chans4, skip_in_channels=chans3, out_channels=chans3)
        self.decoder_stage3 = DecoderStage(in_channels=chans3, skip_in_channels=chans2, out_channels=chans2)
        self.decoder_stage4 = DecoderStage(in_channels=chans2, skip_in_channels=chans1, out_channels=chans1)
        self.output_layer = nn.Sequential(*(Decoder(in_channels=chans1, out_channels=chans1),
                                            nn.Conv2D(in_channels=chans1, out_channels=chans0, kernel_size=3,
                                                      padding=1)))

    def forward(self, img):
        pre = self.conv(img)
        pre = self.relu(pre)
        first = self.encoder_stage1(pre)
        second = self.encoder_stage2(first)
        med = self.encoder_stage3(second)

        med = self.enc2dec(med)
        med = self.med_activate(med)

        de_second = self.decoder_stage2((med, second))
        de_thrid = self.decoder_stage3((de_second, first))
        de_fourth = self.decoder_stage4((de_thrid, pre))

        output = self.output_layer(de_fourth)
        output = output + img
        return output

