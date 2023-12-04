import torch
from torch import nn
# adopted from
# https://github.com/yfsong0709/EfficientGCNv1


class AttentionLayer(nn.Module):
    def __init__(self, out_channel, att_lay, act, **kwargs):
        super(AttentionLayer, self).__init__()

        __attention = {
            'stja': STJointAtt,
            'pa': PartAtt,
            'ca': ChannelAtt,
            'fa': FrameAtt,
            'ja': JointAtt,
            'full': OwnAttention
        }

        self.att = __attention[att_lay](channel=out_channel, **kwargs)
        self.bn = nn.BatchNorm2d(out_channel)
        self.act = act

    def forward(self, x):
        res = x
        x = x * self.att(x)
        return self.act(self.bn(x) + res)


class STJointAtt(nn.Module):
    def __init__(self, channel, reduct_ratio, bias, **kwargs):
        super(STJointAtt, self).__init__()

        inner_channel = int(channel // reduct_ratio)

        self.fcn = nn.Sequential(
            nn.Conv2d(channel, inner_channel, kernel_size=1, bias=bias),
            nn.BatchNorm2d(inner_channel),
            nn.Hardswish(),
            # nn.ReLU(inplace=True)
        )
        self.conv_t = nn.Conv2d(inner_channel, channel, kernel_size=1)
        self.conv_v = nn.Conv2d(inner_channel, channel, kernel_size=1)

    def forward(self, x):
        N, C, T, V = x.size()
        x_t = x.mean(3, keepdims=True)
        x_v = x.mean(2, keepdims=True).transpose(2, 3)
        x_att = self.fcn(torch.cat([x_t, x_v], dim=2))
        x_t, x_v = torch.split(x_att, [T, V], dim=2)
        x_t_att = self.conv_t(x_t).sigmoid()
        x_v_att = self.conv_v(x_v.transpose(2, 3)).sigmoid()
        x_att = x_t_att * x_v_att
        return x_att


class OwnAttention(nn.Module):
    def __init__(self, channel, parts, reduct_ratio, bias, **kwargs):
        super(OwnAttention, self).__init__()

        self.parts = parts
        self.joints = nn.Parameter(self.get_corr_joints(), requires_grad=False)

        inner_channel = int(channel // reduct_ratio)

        self.softmax = nn.Softmax(dim=3)
        self.fcn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, inner_channel, kernel_size=1, bias=bias),
            nn.BatchNorm2d(inner_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channel, channel * len(self.parts), kernel_size=1, bias=bias),
        )

        self.fcn_st = nn.Sequential(
            nn.Conv2d(channel, inner_channel, kernel_size=1, bias=bias),
            nn.BatchNorm2d(inner_channel),
            nn.Hardswish(),
        )

        self.conv_t = nn.Conv2d(inner_channel, channel, kernel_size=1)
        self.conv_v = nn.Conv2d(inner_channel, channel, kernel_size=1)

    def forward(self, x):
        N, C, T, V = x.size()
        # st joint
        x_t = x.mean(3, keepdims=True)
        x_v = x.mean(2, keepdims=True).transpose(2, 3)
        x_att = self.fcn_st(torch.cat([x_t, x_v], dim=2))
        x_t, x_v = torch.split(x_att, [T, V], dim=2)
        x_t_att = self.conv_t(x_t).sigmoid()
        x_v_att = self.conv_v(x_v.transpose(2, 3)).sigmoid()
        x_att_st = x_t_att * x_v_att

        # joint attention
        x_att_joint = self.softmax(self.fcn(x_att_st).view(N, C, 1, len(self.parts)))
        x_att_res = x_att_joint.index_select(3, self.joints).expand_as(x)

        return x_att_res

    def get_corr_joints(self):
        num_joints = sum([len(part) for part in self.parts])
        joints = [j for i in range(num_joints) for j in range(len(self.parts)) if i in self.parts[j]]
        return torch.LongTensor(joints)


class PartAtt(nn.Module):
    def __init__(self, channel, parts, reduct_ratio, bias, **kwargs):
        super(PartAtt, self).__init__()

        self.parts = parts
        self.joints = nn.Parameter(self.get_corr_joints(), requires_grad=False)
        inner_channel = int(channel // reduct_ratio)

        self.softmax = nn.Softmax(dim=3)
        self.fcn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, inner_channel, kernel_size=1, bias=bias),
            nn.BatchNorm2d(inner_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channel, channel * len(self.parts), kernel_size=1, bias=bias),
        )

    def forward(self, x):
        N, C, T, V = x.size()
        x_att = self.softmax(self.fcn(x).view(N, C, 1, len(self.parts)))
        x_att = x_att.index_select(3, self.joints).expand_as(x)
        return x_att

    def get_corr_joints(self):
        num_joints = sum([len(part) for part in self.parts])
        joints = [j for i in range(num_joints) for j in range(len(self.parts)) if i in self.parts[j]]
        return torch.LongTensor(joints)


class ChannelAtt(nn.Module):
    def __init__(self, channel, **kwargs):
        super(ChannelAtt, self).__init__()

        self.fcn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // 4, kernel_size=1),
            nn.BatchNorm2d(channel // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 4, channel, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.fcn(x)


class FrameAtt(nn.Module):
    def __init__(self, **kwargs):
        super(FrameAtt, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv2d(2, 1, kernel_size=(9, 1), padding=(4, 0))

    def forward(self, x):
        x = x.transpose(1, 2)
        x = torch.cat([self.avg_pool(x), self.max_pool(x)], dim=2).transpose(1, 2)
        return self.conv(x)


class JointAtt(nn.Module):
    def __init__(self, parts, **kwargs):
        super(JointAtt, self).__init__()

        num_joint = sum([len(part) for part in parts])

        self.fcn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_joint, num_joint // 2, kernel_size=1),
            nn.BatchNorm2d(num_joint // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_joint // 2, num_joint, kernel_size=1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.fcn(x.transpose(1, 3)).transpose(1, 3)
