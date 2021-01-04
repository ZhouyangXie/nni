from unittest import TestCase, main

import torch

from nni.compression.torch import L2FilterPruner
from nni.compression.torch.speedup import ModelSpeedup

class TorchModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(3, 10, 1, 1)
        self.conv1 = torch.nn.Conv2d(3, 20, 1, 1)
        self.conv2 = torch.nn.Conv2d(30, 2, 1, 1)

    def forward(self, x):
        x0 = self.conv0(x)
        x1 = self.conv1(x)
        x2 = torch.cat((x0, x1), dim=1)
        x3 = self.conv2(x2)
        return x3


class CompressorTestCase(TestCase):
    def test_speedup_ops(self):
        dummy_input = torch.rand(1, 3, 10, 10)
        model = TorchModel()
        _ = model(dummy_input)
        pruner = L2FilterPruner(
            model,
            [{'sparsity': 0.7, 'op_names': ['conv0'], 'op_types': ['Conv2d']}]
        )
        pruner.compress()
        pruner.export_model(model_path='temp_model.pth', mask_path='temp_mask.pth')
        model = TorchModel()
        model.load_state_dict(torch.load('temp_model.pth'))
        ModelSpeedup(model, dummy_input, 'temp_mask.pth', 'cpu').speedup_model()
        _ = model(dummy_input)
       
if __name__ == '__main__':
    main()
