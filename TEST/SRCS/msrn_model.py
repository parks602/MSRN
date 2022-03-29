import torch
import torch.nn as nn



# --------------------------MSRB------------------------------- #

class MSRB_Block(nn.Module):
    def __init__(self):
        super(MSRB_Block, self).__init__()

        channel = 64

        self.conv_3_1 = nn.Conv2d(in_channels = channel, out_channels = channel, kernel_size = 3, stride = 1, padding = 1, bias = True)
        self.conv_3_2 = nn.Conv2d(in_channels = channel * 2, out_channels = channel * 2, kernel_size = 3, stride = 1, padding = 1, bias = True)
        self.conv_5_1 = nn.Conv2d(in_channels = channel, out_channels = channel, kernel_size = 5, stride = 1, padding = 2, bias = True)
        self.conv_5_2 = nn.Conv2d(in_channels = channel * 2, out_channels = channel * 2, kernel_size = 5, stride = 1, padding = 2, bias = True)
        self.confusion = nn.Conv2d(in_channels = channel * 4, out_channels = channel, kernel_size = 1, stride = 1, padding = 0, bias = True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity_data = x
        output_3_1 = self.relu(self.conv_3_1(x))
        output_5_1 = self.relu(self.conv_5_1(x))

        input_2 = torch.cat([output_3_1, output_5_1], 1)
        output_3_2 = self.relu(self.conv_3_2(input_2))
        output_5_2 = self.relu(self.conv_5_2(input_2))
        output = torch.cat([output_3_2, output_5_2], 1)
        output = self.confusion(output)
        output = torch.add(output, identity_data)
        return output

# --------------------------MSRN------------------------------- #

class MSRN(nn.Module):
    def __init__(self):
        super(MSRN, self).__init__()

        # channel = out_channels_MSRB
        out_channels_MSRB = 64
        scale = 5

        self.conv_input = nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size = 3, stride = 1, padding = 1, bias = True)
        self.conv_input2 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1, bias = True)
        self.residual1 = self.make_layer(MSRB_Block)
        self.residual2 = self.make_layer(MSRB_Block)
        self.residual3 = self.make_layer(MSRB_Block)
        self.residual4 = self.make_layer(MSRB_Block)
        self.residual5 = self.make_layer(MSRB_Block)
        self.residual6 = self.make_layer(MSRB_Block)
        self.residual7 = self.make_layer(MSRB_Block)
        self.residual8 = self.make_layer(MSRB_Block)
        self.bottle = nn.Conv2d(in_channels= out_channels_MSRB * 8 + 64, out_channels = 64, kernel_size = 1, stride = 1, padding = 0, bias = True)
        self.geo_bottle = nn.Conv2d(in_channels= out_channels_MSRB * 5, out_channels = 64, kernel_size = 1, stride = 1, padding = 0, bias = True)
        self.conv = nn.Conv2d(in_channels = 64, out_channels = 64 * scale * scale, kernel_size = 3, stride = 1, padding = 1, bias = True)
        self.geo_conv = nn.Conv2d(in_channels = 192, out_channels = 64, kernel_size = 3, stride = 1, padding = 1, bias = True)
        self.new_conv = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1, bias = True)
        self.convt = nn.PixelShuffle(scale)
        self.conv_output = nn.Conv2d(in_channels = 64, out_channels = 1, kernel_size = 3, stride = 1, padding = 1, bias = True)


    def make_layer(self, block):
        layers = []
        layers.append(block())
        return nn.Sequential(*layers)
    def forward(self, x, geomap, gismap):
        out = self.conv_input(x)
        LR = out
        out = self.residual1(out)
        concat1 = out
        out = self.residual2(out)
        concat2 = out
        out = self.residual3(out)
        concat3 = out
        out = self.residual4(out)
        concat4 = out
        out = self.residual5(out)
        concat5 = out
        out = self.residual6(out)
        concat6 = out
        out = self.residual7(out)
        concat7 = out
        out = self.residual8(out)
        concat8 = out

        geo = self.conv_input2(geomap)
        geo1 = self.residual1(geo)
        geo2 = self.residual2(geo1)
        geo3 = self.residual3(geo2)
        geo4 = self.residual4(geo3)

        gis = self.conv_input2(gismap)
        gis1 = self.residual1(gis)
        gis2 = self.residual2(gis1)
        gis3 = self.residual3(gis2)
        gis4 = self.residual4(gis3)

        out = torch.cat([LR, concat1, concat2, concat3, concat4, concat5, concat6, concat7, concat8], 1)
        out = self.bottle(out)
        out = self.convt(self.conv(out))
        out_geo = torch.cat([geo, geo1, geo2, geo3, geo4],1)
        out_geo = self.geo_bottle(out_geo)

        out_gis = torch.cat([gis, gis1, gis2, gis3, gis4],1)
        out_gis = self.geo_bottle(out_gis)

        out = torch.cat([out, out_geo, out_gis],1)
        #out = torch.cat([out, geo], 1)
        out = self.geo_conv(out)
        out = self.new_conv(out)
        out = self.new_conv(out)
        out = self.new_conv(out)
        out = self.conv_output(out)
        return out

