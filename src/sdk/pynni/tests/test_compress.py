import torch
import torch.nn as nn
import torch.nn.functional as F

from nni.compression.torch import L1FilterPruner
from nni.compression.torch.speedup import ModelSpeedup


class ConvBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 10, 3)
        self.bn = nn.BatchNorm2d(10)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x


def test_compress_speedup():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    pruner = L1FilterPruner(
        model=ConvBlock(),
        config_list=[{
            'sparsity': 0.5,
            'op_types': ['Conv2d']}]
        )
    model = pruner.compress()

    model = model.to(device)
    dummy_input = torch.rand(1, 1, 11, 11).to(device)

    y_compress = model(dummy_input).to(device)

    pruner.export_model(model_path='finetuned.pth', mask_path='mask.pth')

    model = ConvBlock().load_state_dict(
        torch.load('finetuned.pth'))
    ModelSpeedup(model, dummy_input, 'mask.pth', device).speedup_model()

    y_speedup = model(dummy_input).to(device)

    assert (y_compress == y_speedup).all()


if __name__ == "__main__":
    test_compress_speedup()
