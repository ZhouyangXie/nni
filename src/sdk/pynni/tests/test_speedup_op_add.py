from unittest import TestCase, main

import torch

from nni.compression.torch import SlimPruner
from nni.compression.torch.speedup import ModelSpeedup

class TorchModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(3, 10, 1, 1)
        self.bn0 = torch.nn.BatchNorm2d(10)
        self.bn0.weight.data = torch.rand(10, requires_grad=True)
        self.conv1 = torch.nn.Conv2d(10, 10, 1, 1)
        self.bn1 = torch.nn.BatchNorm2d(10)
        self.bn1.weight.data = torch.rand(10, requires_grad=True)

    def forward(self, x):
        x = self.conv0(x)
        x0 = self.bn0(x)
        x1 = self.conv1(x0)
        x1 = self.bn1(x1)
        return x0 + x1


class CompressorTestCase(TestCase):
    def test_speedup_ops(self):
        dummy_input = torch.rand(1, 3, 10, 10)
        model = TorchModel()
        _ = model(dummy_input)
        pruner = SlimPruner(
            model,
            [{
            'sparsity': 0.1,
            'op_names': ['bn0', 'bn1'],
            'op_types': ['BatchNorm2d'],
            }]
        )
        pruner.compress()
        pruner.export_model(model_path='temp_model.pth', mask_path='temp_mask.pth')
        model = TorchModel()
        model.load_state_dict(torch.load('temp_model.pth'))
        ModelSpeedup(model, dummy_input, 'temp_mask.pth', 'cpu').speedup_model()
        _ = model(dummy_input)
       
if __name__ == '__main__':
    CompressorTestCase().test_speedup_ops()
