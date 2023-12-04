import torch
from torch import nn

from src.model.layers import Classifier, InputStream, MainStream


class Student(nn.Module):
    def __init__(self, actions, arch_choices, student_id, args, load=False, **kwargs):
        super(Student, self).__init__()
        self.student_id = student_id
        self.args = args
        self.kwargs = kwargs
        self.num_input, self.num_channel, _, _, _ = kwargs.get('data_shape')

        # skeleton
        self.input_stream = None
        self.main_stream = None
        self.classifier = None

        self.action_list = dict.fromkeys(arch_choices)
        computations = list(arch_choices.keys())
        for comp, action in zip(computations, actions):
            self.action_list[comp] = arch_choices[comp][action]

        # make sure all actions are set
        assert None not in self.action_list

        self.__build_student()

    def __build_student(self):
        # build input stream
        # depending on the input size -> JVB == 3 --> JVBA == 4
        # bug in torch so this has to be initialized before!
        input_stream_helper = InputStream(self.action_list, self.num_channel, self.args, **self.kwargs)

        self.input_stream = nn.ModuleList([InputStream(self.action_list, self.num_channel, self.args, **self.kwargs)
                        for _ in range(self.num_input)])

        # build mainstream
        input_main = input_stream_helper.last_channel * self.num_input
        self.main_stream = MainStream(self.action_list, input_main, self.args, **self.kwargs)

        # build classifier
        input_classifier = self.main_stream.last_channel
        drop_prob = self.action_list.get("drop_prob")
        self.classifier = Classifier(input_classifier, drop_prob, self.args.old_sp, **self.kwargs)

        # init parameters
        self.__init_param(self.modules())

    def forward(self, x):
        N, I, C, T, V, M = x.size()
        x = x.permute(1, 0, 5, 2, 3, 4).contiguous().view(I, N*M, C, T, V)

        # input branches
        x = torch.cat([branch(x[i]) for i, branch in enumerate(self.input_stream)], dim=1)

        # main stream
        x = self.main_stream(x)

        # output
        _, C, T, V = x.size()
        feature = x.view(N, M, C, T, V).permute(0, 2, 3, 4, 1)
        if self.args.old_sp:
            out = self.classifier(feature).view(N, -1)
        else:
            c_new = x.size(1)
            x = x.view(N, M, c_new, -1)
            x = x.mean(3).mean(1)

            out = self.classifier(x)
        # 1, 54000
        return out, feature

    def forward_before_global_avg_pool(self, x):
        outputs = []
        def hook_fn(module, input_t, output_t):
            outputs.append(output_t)

        for m in self.modules():
            if isinstance(m, torch.nn.AdaptiveAvgPool2d):
                m.register_forward_hook(hook_fn)
        self.forward(x)

        assert len(outputs) == 1, f"Expected 1 AdaptiveAvgPool2d, got {len(outputs)}"
        return outputs[0]

    @staticmethod
    def __init_param(modules):
        for m in modules:
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
