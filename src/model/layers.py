import logging
import sys

from src.model.attention_layers import AttentionLayer
from src.model.activation_layers import *

# adopted from
# https://github.com/yfsong0709/EfficientGCNv1


class Stream(nn.Sequential):

    def __init__(self, actions, args, **kwargs):
        super(Stream, self).__init__()

        self.args = args

        self.act_poss = {
            "relu": nn.ReLU(inplace=True),
            "relu6": nn.ReLU6(inplace=True),
            "silu": nn.SiLU(inplace=True),
            "hardswish": nn.Hardswish(inplace=True),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'swish': Swish(inplace=True),
            'acon': AconC(channel=8),
            'meta': MetaAconC(channel=8)
        }

        if self.args.old_sp:
            self.init_lay = actions.get("init_lay")
            self.temp_win_in = actions.get("temp_win")
            self.graph_dist_in = actions.get("graph_dist")
            self.conv_layer = actions.get("conv_lay")
            self.blocks_in = actions.get("blocks_in")
            self.depth_in = actions.get("depth_in")
            self.stride_in = actions.get("stride_in")
            self.blocks_main = actions.get("blocks_main")
            self.depth_main = actions.get("depth_main")
            self.stride_main = actions.get("stride_main")
            self.reduct_ratio_in = actions.get("reduct_ratio")
            self.expand_ratio = actions.get("expand_ratio")
            self.att_lay = actions.get("att_lay")
            self.last_channel = self.init_lay
            self.temporal_layer = None
        else:
            # common
            self.init_lay = actions.get("init_lay")
            self.conv_layer = actions.get("conv_lay")
            self.expand_ratio = 0
            self.att_lay = actions.get("att_lay")
            self.last_channel = self.init_lay
            self.temporal_layer = None
            # input stream
            self.blocks_in = actions.get("blocks_in")
            self.depth_in = actions.get("depth_in")
            self.temp_win_in = actions.get("temp_win_in")
            self.graph_dist_in = actions.get("graph_dist_in")
            self.stride_in = actions.get("stride_in")
            self.reduct_ratio_in = actions.get("reduct_ratio_in")
            # mainstream
            self.blocks_main = actions.get("blocks_main")
            self.depth_main = actions.get("depth_main")
            self.graph_dist_main = actions.get("graph_dist_main")
            self.shrinkage_main = actions.get("shrinkage_main")
            self.residual_main = actions.get("residual_main")
            self.adaptive_main = actions.get("adaptive_main")

        try:
            # import temporal layer class
            self.temporal_layer = getattr(sys.modules[__name__], f'Temporal_{self.conv_layer}_Layer')
        except:
            logging.error("The conv layer: {} is not known!".format(self.conv_layer))

        self.kwargs = kwargs
        self.kwargs["act"] = self.act_poss.get(actions.get("act"))
        self.kwargs["expand_ratio"] = self.expand_ratio
        self.kwargs["reduct_ratio"] = self.reduct_ratio_in

        # add channel scaler make it working with prev. impl.
        try:
            self.scale_in = actions.get("scale_in")
            self.scale_main = actions.get("scale_main")
        except:
            self.scale_in = 0.5
            self.scale_main = 2
            logging.info("Scaling not given in search space -> setting value to: {}/{}".format(self.scale_in,
                                                                                                      self.scale_main))


class InputStream(Stream):

    def __init__(self, actions, num_channel, args, **kwargs):
        super(InputStream, self).__init__(actions, args, **kwargs)

        self.args = args

        # fixed starting layer
        self.add_module('init_bn', nn.BatchNorm2d(num_channel))
        self.add_module('stem_scn', Spatial_Graph_Layer(num_channel, self.init_lay, self.graph_dist_in, **self.kwargs))
        self.add_module('stem_tcn', Temporal_Basic_Layer(self.init_lay, self.temp_win_in, **self.kwargs))

        for i in range(self.blocks_in):
            # min to 8? or bigger
            channel = max(int(round(self.last_channel * self.scale_in / 16)) * 16, 32)
            # channel = round(int(self.last_channel / self.reduct_ratio))
            self.add_module(f'block-{i}_scn', Spatial_Graph_Layer(self.last_channel, channel, self.graph_dist_in,
                                                                  **self.kwargs))
            for j in range(self.depth_in):
                s = self.stride_in if j == 0 else 1
                self.add_module(f'block-{i}_tcn-{j}', self.temporal_layer(channel, self.temp_win_in, stride=s,
                                                                          **self.kwargs))
            self.add_module(f'block-{i}_att', AttentionLayer(channel, self.att_lay, **self.kwargs))
            self.last_channel = channel

    @property
    def last_channel(self):
        return self._last_channel

    @last_channel.setter
    def last_channel(self, val):
        self._last_channel = val


class MainStream(Stream):

    def __init__(self, actions, input_main, args, **kwargs):
        super(MainStream, self).__init__(actions, args, **kwargs)

        self.args = args

        self.last_channel = input_main

        if self.args.old_sp:
            for i in range(self.blocks_main):
                channel = max(int(round(self.last_channel * self.scale_main / 16)) * 16, 32)
                # channel = int(self.last_channel * self.expand_ratio)
                self.add_module(f'block-{i}_scn', Spatial_Graph_Layer(self.last_channel, channel, self.graph_dist_in,
                                                                      **self.kwargs))
                for j in range(self.depth_main):
                    s = self.stride_main if j == 0 else 1
                    self.add_module(f'block-{i}_tcn-{j}', self.temporal_layer(channel, self.temp_win_in, stride=s,
                                                                              **self.kwargs))
                self.add_module(f'block-{i}_att', AttentionLayer(channel, self.att_lay, **self.kwargs))
                self.last_channel = channel
        else:
            # calculate channels
            channels_in = [self.last_channel, 128, 128, 256, 256, 256, 320, 320]
            channels_out = [128, 128, 256, 256, 256, 320, 320, 320]
            for idx, i in enumerate(range(0, self.blocks_main)):
                cur_channel_in = channels_in[idx]
                cur_channel_out = channels_out[idx]
                # gcn -> attention already included
                self.add_module(f'block-{i}_gcn_tcn_main', TcnGcnUnit(cur_channel_in, cur_channel_out, self.shrinkage_main,
                                                             self.depth_main, **self.kwargs))

            self.last_channel = cur_channel_out

    @property
    def last_channel(self):
        return self._last_channel

    @last_channel.setter
    def last_channel(self, val):
        self._last_channel = val


class TcnGcnUnit(nn.Module):
    def __init__(self, in_channels, out_channels, shrinkage, depth_main, A, act, stride=1, residual=True, adaptive=True,
                 attention=True, **kwargs):
        super(TcnGcnUnit, self).__init__()
        self.gcn1 = GCNMain(in_channels, out_channels, A, act, shrinkage, depth_main,
                            adaptive=adaptive, attention=attention, **kwargs)
        self.tcn1 = TCNMain(out_channels, out_channels, stride=stride)
        self.act = act
        self.attention = attention

        if not residual:
            self.residual = ZeroLayer()

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = nn.Identity()

        else:
            self.residual = TCNMain(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.act(self.tcn1(self.gcn1(x)) + self.residual(x))
        return x


class GCNMain(nn.Module):
    def __init__(self, in_channels, out_channels, A, act, shrinkage, depth_main, adaptive, attention=True, **kwargs):
        super(GCNMain, self).__init__()
        inter_channels = out_channels // shrinkage
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.depth_main = depth_main
        num_jpts = A.shape[-1]

        self.conv_d = nn.ModuleList()
        for i in range(self.depth_main):
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if adaptive:
            self.PA = nn.Parameter(A, requires_grad=False)
            self.alpha = nn.Parameter(torch.zeros(1))
            self.conv_a = nn.ModuleList()
            self.conv_b = nn.ModuleList()
            for i in range(self.depth_main):
                self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))
                self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
        else:
            self.A = nn.Parameter(A, requires_grad=False)
        self.adaptive = adaptive

        if attention:
            # temporal attention
            self.conv_ta = nn.Conv1d(out_channels, 1, 9, padding=4)

            # s attention
            ker_jpt = num_jpts - 1 if not num_jpts % 2 else num_jpts
            pad = (ker_jpt - 1) // 2
            self.conv_sa = nn.Conv1d(out_channels, 1, ker_jpt, padding=pad)

            # channel attention
            rr = 2
            self.fc1c = nn.Linear(out_channels, out_channels // rr)
            self.fc2c = nn.Linear(out_channels // rr, out_channels)

        self.attention = attention

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.tan = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.act = act

    def forward(self, x):
        N, C, T, V = x.size()

        y = None
        if self.adaptive:
            A = self.PA
            # A = A + self.PA
            for i in range(self.depth_main):
                A1 = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)
                A2 = self.conv_b[i](x).view(N, self.inter_c * T, V)
                A1 = self.tan(torch.matmul(A1, A2) / A1.size(-1))  # N V V
                A1 = A[i] + A1 * self.alpha
                A2 = x.view(N, C * T, V)
                z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
                y = z + y if y is not None else z
        else:
            A = self.A.cuda(x.get_device()) * self.mask
            for i in range(self.num_subset):
                A1 = A[i]
                A2 = x.view(N, C * T, V)
                z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
                y = z + y if y is not None else z

        y = self.act(y + self.down(x) + self.bn(y))

        if self.attention:
            # spatial attention
            se = y.mean(-2)  # N C V
            se1 = self.sigmoid(self.conv_sa(se))
            y = y * se1.unsqueeze(-2) + y
            # temporal attention
            se = y.mean(-1)
            se1 = self.sigmoid(self.conv_ta(se))
            y = y * se1.unsqueeze(-1) + y
            # channel attention
            se = y.mean(-1).mean(-1)
            se1 = self.act(self.fc1c(se))
            se2 = self.sigmoid(self.fc2c(se1))
            y = y * se2.unsqueeze(-1).unsqueeze(-1) + y
        return y


class TCNMain(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(TCNMain, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # TODO also activation here??
        x = self.bn(self.conv(x))
        return x


class BasicLayer(nn.Module):
    def __init__(self, in_channel, out_channel, residual, bias, act, **kwargs):
        super(BasicLayer, self).__init__()

        self.conv = nn.Conv2d(in_channel, out_channel, 1, bias=bias)
        self.bn = nn.BatchNorm2d(out_channel)

        self.residual = nn.Identity() if residual else ZeroLayer()
        self.act = act

    def forward(self, x):
        res = self.residual(x)
        x = self.act(self.bn(self.conv(x)) + res)
        return x


class Spatial_Graph_Layer(BasicLayer):
    def __init__(self, in_channel, out_channel, max_graph_distance, bias, residual=True, **kwargs):
        super(Spatial_Graph_Layer, self).__init__(in_channel, out_channel, residual, bias, **kwargs)

        self.conv = SpatialGraphConv(in_channel, out_channel, max_graph_distance, bias, **kwargs)
        if residual and in_channel != out_channel:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 1, bias=bias),
                nn.BatchNorm2d(out_channel),
            )


class Temporal_Basic_Layer(BasicLayer):
    def __init__(self, channel, temporal_window_size, bias, stride=1, residual=True, **kwargs):
        super(Temporal_Basic_Layer, self).__init__(channel, channel, residual, bias, **kwargs)

        padding = (temporal_window_size - 1) // 2
        self.conv = nn.Conv2d(channel, channel, (temporal_window_size,1), (stride,1), (padding,0), bias=bias)
        if residual and stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(channel, channel, 1, (stride,1), bias=bias),
                nn.BatchNorm2d(channel),
            )


class Temporal_Bottleneck_Layer(nn.Module):
    def __init__(self, channel, temporal_window_size, bias, act, reduct_ratio, stride=1, residual=True, **kwargs):
        super(Temporal_Bottleneck_Layer, self).__init__()

        inner_channel = int(channel // reduct_ratio)
        padding = (temporal_window_size - 1) // 2
        self.act = act

        self.reduct_conv = nn.Sequential(
            nn.Conv2d(channel, inner_channel, 1, bias=bias),
            nn.BatchNorm2d(inner_channel),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(inner_channel, inner_channel, (temporal_window_size,1), (stride,1), (padding,0), bias=bias),
            nn.BatchNorm2d(inner_channel),
        )
        self.expand_conv = nn.Sequential(
            nn.Conv2d(inner_channel, channel, 1, bias=bias),
            nn.BatchNorm2d(channel),
        )

        if not residual:
            self.residual = ZeroLayer()
        elif stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(channel, channel, 1, (stride,1), bias=bias),
                nn.BatchNorm2d(channel),
            )

    def forward(self, x):
        res = self.residual(x)
        x = self.act(self.reduct_conv(x))
        x = self.act(self.conv(x))
        x = self.act(self.expand_conv(x) + res)
        return x


class Temporal_Sep_Layer(nn.Module):
    def __init__(self, channel, temporal_window_size, bias, act, expand_ratio, stride=1, residual=True, **kwargs):
        super(Temporal_Sep_Layer, self).__init__()

        padding = (temporal_window_size - 1) // 2
        self.act = act

        if expand_ratio > 0:
            inner_channel = int(channel * expand_ratio)
            self.expand_conv = nn.Sequential(
                nn.Conv2d(channel, inner_channel, 1, bias=bias),
                nn.BatchNorm2d(inner_channel),
            )
        else:
            inner_channel = channel
            self.expand_conv = None

        self.depth_conv = nn.Sequential(
            nn.Conv2d(inner_channel, inner_channel, (temporal_window_size,1), (stride,1), (padding,0), groups=inner_channel, bias=bias),
            nn.BatchNorm2d(inner_channel),
        )
        self.point_conv = nn.Sequential(
            nn.Conv2d(inner_channel, channel, 1, bias=bias),
            nn.BatchNorm2d(channel),
        )
        if not residual:
            self.residual = ZeroLayer()
        elif stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(channel, channel, 1, (stride,1), bias=bias),
                nn.BatchNorm2d(channel),
            )

    def forward(self, x):
        res = self.residual(x)
        if self.expand_conv is not None:
            x = self.act(self.expand_conv(x))
        x = self.act(self.depth_conv(x))
        x = self.point_conv(x)
        return x + res


class Temporal_SG_Layer(nn.Module):
    def __init__(self, channel, temporal_window_size, bias, act, reduct_ratio, stride=1, residual=True, **kwargs):
        super(Temporal_SG_Layer, self).__init__()

        padding = (temporal_window_size - 1) // 2
        inner_channel = int(channel // reduct_ratio)
        self.act = act

        self.depth_conv1 = nn.Sequential(
            nn.Conv2d(channel, channel, (temporal_window_size,1), 1, (padding,0), groups=channel, bias=bias),
            nn.BatchNorm2d(channel),
        )
        self.point_conv1 = nn.Sequential(
            nn.Conv2d(channel, inner_channel, 1, bias=bias),
            nn.BatchNorm2d(inner_channel),
        )
        self.point_conv2 = nn.Sequential(
            nn.Conv2d(inner_channel, channel, 1, bias=bias),
            nn.BatchNorm2d(channel),
        )
        self.depth_conv2 = nn.Sequential(
            nn.Conv2d(channel, channel, (temporal_window_size,1), (stride,1), (padding,0), groups=channel, bias=bias),
            nn.BatchNorm2d(channel),
        )

        if not residual:
            self.residual = ZeroLayer()
        elif stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(channel, channel, 1, (stride,1), bias=bias),
                nn.BatchNorm2d(channel),
            )

    def forward(self, x):
        res = self.residual(x)
        x = self.act(self.depth_conv1(x))
        x = self.point_conv1(x)
        x = self.act(self.point_conv2(x))
        x = self.depth_conv2(x)
        return x + res


class Temporal_V3_Layer(nn.Module):
    def __init__(self, channel, temporal_window_size, bias, act, expand_ratio, stride=1, residual=True,
                 squeez_excite=True, reduct=4,  **kwargs):
        super(Temporal_V3_Layer, self).__init__()

        padding = (temporal_window_size - 1) // 2
        # self.act = act
        self.act = nn.Hardswish(inplace=True)

        if expand_ratio > 0:
            inner_channel = int(channel * expand_ratio)
            self.expand_conv = nn.Sequential(
                nn.Conv2d(channel, inner_channel, 1, bias=bias),
                nn.BatchNorm2d(inner_channel),
            )
        else:
            inner_channel = channel
            self.expand_conv = None

        self.depth_conv = nn.Sequential(
            nn.Conv2d(inner_channel, inner_channel, (temporal_window_size, 1), (stride, 1), (padding, 0),
                      groups=inner_channel, bias=bias),
            nn.BatchNorm2d(inner_channel),
        )

        if squeez_excite:
            self.squeez_excite = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(inner_channel, inner_channel // reduct, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(inner_channel // reduct),
                nn.ReLU(inplace=True),
                nn.Conv2d(inner_channel // reduct, inner_channel, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(inner_channel),
                nn.Hardswish(inplace=True)
            )
        else:
            self.squeez_excite = nn.Identity()

        self.point_conv = nn.Sequential(
            nn.Conv2d(inner_channel, channel, 1, bias=bias),
            nn.BatchNorm2d(channel),
        )

        if not residual:
            self.residual = ZeroLayer()
        elif stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(channel, channel, 1, (stride, 1), bias=bias),
                nn.BatchNorm2d(channel),
            )

    def forward(self, x):
        res = self.residual(x)
        if self.expand_conv is not None:
            x = self.act(self.expand_conv(x))
        x = self.depth_conv(x)
        x = self.squeez_excite(x)
        x = self.act(self.point_conv(x))
        return x + res


class Temporal_Shuffle_Layer(nn.Module):
    """
    ShuffleNet with pointwise group
    Point group conv + Channel shuffle + 3x3 depth conv + point group conv and residual
    """
    def __init__(self, channel, temporal_window_size, bias, act, reduct_ratio, stride=1, residual=True, combine=False,
                 **kwargs):
        super(Temporal_Shuffle_Layer, self).__init__()

        padding = (temporal_window_size - 1) // 2
        # in paper // 4
        inner_channel = int(channel // reduct_ratio)
        self.act = act
        self.combine = combine

        self.groups = inner_channel

        # no group conv
        self.point_conv = nn.Sequential(
            nn.Conv2d(channel, inner_channel, 1, bias=bias),
            nn.BatchNorm2d(inner_channel),
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # in paper 3x3 kernel and stride == 2
        # try with other depth
        self.depth_conv = nn.Sequential(
            # nn.Conv2d(inner_channel, inner_channel, (3, 3), (stride, 1), (padding, 0),
            #            groups=inner_channel, bias=bias),
            nn.Conv2d(inner_channel, inner_channel, (temporal_window_size, 1), (stride, 1), (padding, 0),
                      groups=inner_channel, bias=bias),
            nn.BatchNorm2d(inner_channel),
        )

        self.point_conv_expand = nn.Sequential(
            nn.Conv2d(inner_channel, channel, 1, bias=bias),
            nn.BatchNorm2d(channel)
        )

        if not residual:
            self.residual = ZeroLayer()
        elif stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(channel, channel, 1, (stride, 1), bias=bias),
                nn.BatchNorm2d(channel),
            )

    @staticmethod
    def channel_shuffle(x, groups):
        batchsize, num_channels, time, vertices = x.data.size()
        channels_per_group = num_channels // groups
        # reshape
        x = x.view(batchsize, groups,
                   channels_per_group, time, vertices)
        x = torch.transpose(x, 1, 2).contiguous()
        # flatten
        x = x.view(batchsize, -1, time, vertices)

        return x

    def forward(self, x):
        res = self.residual(x)
        if self.combine:
            res = F.avg_pool3d(x, kernel_size=3, stride=2, padding=1)

        x = self.point_conv(x)
        x = self.channel_shuffle(x, self.groups)
        x = self.depth_conv(x)
        x = self.point_conv_expand(x)
        if self.combine:
            x = torch.cat((res, x), -1)
        else:
            x += res
        return self.act(x)


class SpatialGraphConv(nn.Module):
    """
    https://github.com/yysijie/st-gcn
    """
    def __init__(self, in_channel, out_channel, max_graph_distance, bias, edge, A, **kwargs):
        super(SpatialGraphConv, self).__init__()

        self.s_kernel_size = max_graph_distance + 1
        self.gcn = nn.Conv2d(in_channel, out_channel*self.s_kernel_size, 1, bias=bias)
        self.A = nn.Parameter(A[:self.s_kernel_size], requires_grad=False)
        if edge:
            self.edge = nn.Parameter(torch.ones_like(self.A))
        else:
            self.edge = 1

    def forward(self, x):
        x = self.gcn(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.s_kernel_size, kc//self.s_kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, self.A * self.edge)).contiguous()
        return x


class Classifier(nn.Sequential):
    def __init__(self, curr_channel, drop_prob, old_sp, num_class, **kwargs):
        super(Classifier, self).__init__()

        if old_sp:
            self.add_module('gap', nn.AdaptiveAvgPool3d(1))
            self.add_module('dropout', nn.Dropout(drop_prob, inplace=True))
            self.add_module('fc', nn.Conv3d(curr_channel, num_class, kernel_size=1))
        else:
            self.add_module('dropout', nn.Dropout(drop_prob, inplace=True))
            self.add_module('fc', nn.Linear(curr_channel, num_class))


class ZeroLayer(nn.Module):
    def __init__(self):
        super(ZeroLayer, self).__init__()

    def forward(self, x):
        return 0
