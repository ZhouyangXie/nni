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


from nni.compression.torch.compressor import Pruner
class NanMasker(object):
    _valid_resolvers = [
        torch.nn.Linear,
        torch.nn.Conv1d,
        torch.nn.Conv2d,
        torch.nn.Conv3d,
        torch.nn.ConvTranspose1d,
        torch.nn.ConvTranspose2d,
        torch.nn.ConvTranspose3d,
    ]

    def __init__(self, pruner) -> None:
        super().__init__()
        assert isinstance(pruner, Pruner)
        self.pruner = pruner
        self._resolver_handles = []

    @staticmethod
    def _hook_resolve_nan(module, inputs):
        for input in inputs:
            if isinstance(input, torch.Tensor):
                input[torch.isnan(input)] = 0

    def __enter__(self):
        self._set_nan_mask()
        self._resolver_handles = []
        for module in self.pruner.bound_model.modules():
            if type(module) in self._valid_resolvers:
                self._resolver_handles.append(
                    module.register_forward_pre_hook(
                        self._hook_resolve_nan
                    )
                )

    def _set_nan_mask(self):
        for wrapper in self.pruner.get_modules_wrapper():
            wrapper.weight_mask[wrapper.weight_mask==0] = float('nan')
            if wrapper.bias_mask is not None:
                wrapper.bias_mask[wrapper.bias_mask==0] = float('nan')

    def __exit__(self, exc_type, exc_val, _):
        self._remove_nan_mask()
        for handle in self._resolver_handles:
            handle.remove()
        self._resolver_handles = []
        if exc_type is not None:
            exc_type(exc_val)

    def _remove_nan_mask(self):
        for wrapper in self.pruner.get_modules_wrapper():
            wrapper.weight_mask[torch.isnan(wrapper.weight_mask)] = 0
            if wrapper.bias_mask is not None:
                wrapper.bias_mask[torch.isnan(wrapper.bias_mask)] = 0

def nan_masking(pruner):
    return NanMasker(pruner)

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

    with nan_masking(pruner):
        model = model.to(device)
        dummy_input = torch.rand(1, 1, 11, 11).to(device)
        y_compress = model(dummy_input)

    pruner.export_model(model_path='finetuned.pth', mask_path='mask.pth')

    model = TwoConvs()
    ds = torch.load('finetuned.pth')
    model.load_state_dict(ds)
    ModelSpeedup(model, dummy_input, 'mask.pth', device).speedup_model()

    y_speedup = model(dummy_input).to(device)

    assert (y_compress == y_speedup).all()


if __name__ == "__main__":
    test_compress_speedup()
