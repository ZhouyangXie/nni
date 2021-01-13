import torch
import torch.nn as nn
import torch.nn.functional as F

from nni.compression.torch import L1FilterPruner
from nni.compression.torch.speedup import ModelSpeedup


class TwoConvs(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(1, 10, 3)
        self.bn0 = nn.BatchNorm2d(10)
        self.conv1 = nn.Conv2d(10, 5, 3)
        self.bn1 = nn.BatchNorm2d(5)

        for param in self.parameters():
            param.data = torch.rand(param.data.shape)

    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        x = F.relu(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        return x


def test_compress_speedup():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    pruner = L1FilterPruner(
        model=TwoConvs(),
        config_list=[{
            'sparsity': 0.5,
            'op_types': ['Conv2d'],
            'op_names': ['conv0']}]
        )
    model = pruner.compress()

    model = model.to(device)
    dummy_input = torch.rand(1, 1, 11, 11).to(device)

    y_compress = model(dummy_input).to(device)

    pruner.export_model(model_path='finetuned.pth', mask_path='mask.pth')

    model = TwoConvs()
    model.load_state_dict(
        torch.load('finetuned.pth'))
    ModelSpeedup(model, dummy_input, 'mask.pth', device).speedup_model()

    y_speedup = model(dummy_input).to(device)

    assert (y_compress != y_speedup).any()


if __name__ == "__main__":
    test_compress_speedup()
