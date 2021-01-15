import torch
from torch.autograd import Function
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module

from nni.compression.torch import L1FilterPruner
from nni.compression.torch.speedup import ModelSpeedup


class TwoConvs(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(1, 10, 3)
        self.bn0 = nn.BatchNorm2d(10)
        self.conv1 = nn.Conv2d(1, 10, 3)
        self.bn1 = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(10, 1, 3)

        for param in self.parameters():
            # re-initialize to avoid all zero initialization
            param.data = torch.rand(param.data.shape)

    def forward(self, x):
        out0 = self.conv0(x)
        out0 = self.bn0(out0)
        out0 = F.relu(out0)

        out1 = self.conv1(x)
        out1 = self.bn1(out1)
        out1 = F.relu(out1)

        out = out0 + out1
        return self.conv2(out)


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
        class resolve_nan(Function):
            def __init__(self) -> None:
                super().__init__()
                self.nan_mask = None
            
            def forward(self, x):
                self.nan_mask = torch.isnan(x)
                x_unmasked = x.clone()
                x_unmasked[self.nan_mask] = 0
                return x_unmasked

            def backward(self, output_grad):
                input_grad = output_grad.clone()
                input_grad[self.nan_mask] = 0
                return input_grad

        return tuple([
            resolve_nan()(input) for input in inputs
        ])

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
    dummy_input = torch.rand(1, 1, 21, 21).to(device)
    pruner = L1FilterPruner(
        model=TwoConvs(),
        config_list=[{
            'sparsity': 0.5,
            'op_types': ['Conv2d'],
            'op_names': ['conv0', 'conv1']}],
        dependency_aware=True,
        dummy_input=dummy_input
        )
    model = pruner.compress()

    with nan_masking(pruner):
        # torch.autograd.set_detect_anomaly(True)
        model = model.to(device)
        model.train()
        dummy_input = torch.rand(1, 1, 11, 11).to(device)
        y_compress = model(dummy_input)
        loss = y_compress.sum()
        loss.backward()

    pruner.export_model(model_path='finetuned.pth', mask_path='mask.pth')

    model = TwoConvs()
    ds = torch.load('finetuned.pth')
    model.load_state_dict(ds)
    ModelSpeedup(model, dummy_input, 'mask.pth', device).speedup_model()

    y_speedup = model(dummy_input).to(device)


    assert not torch.isnan(y_speedup).any()
    assert (y_compress == y_speedup).all()


if __name__ == "__main__":
    test_compress_speedup()
