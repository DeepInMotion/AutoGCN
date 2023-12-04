import torch.nn


class Policy(torch.nn.Module):
    def __init__(self, size, device, initial_params=None):
        super(Policy, self).__init__()

        if initial_params is not None:
            self.params = torch.nn.Parameter(torch.tensor(initial_params, device=device))
        else:
            assert size >= 1
            self.params = torch.nn.Parameter(torch.ones((size,), device=device))

    def forward(self):
        return torch.nn.Identity()(self.params)
