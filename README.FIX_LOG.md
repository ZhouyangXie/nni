# Notes for fixes over branch ubd
This is a note for the fixes over NNI(v1.9) on this new branch ubd. Trace the changes in files by 'git log'. We skip some trivial commits whose message is self-explanatory.

* commit 84d2503334d3c82e69a4a780b5af5f4c23f4c283 (HEAD -> ubd)
Author: Jerry Xie <oceania.xie@gmail.com>
Date:   Fri Jan 15 12:22:58 2021 +0800

    add op can trigger NaN result without dependency awareness

    We modify the test `/src/sdk/pynni/tests/test_compress.py` with `add` operation of tensors. This test shows that if the model is not pruned with `dependency_aware` mode, SpeedUp with keep some masked channels of convs before `add`. It will trigger NaN result in speedup models due to our NaN masking strategy.

* commit e2b00fe72d167e6e815ff9ef4c498f17b779ba82
Author: Jerry Xie <oceania.xie@gmail.com>
Date:   Wed Jan 13 16:53:08 2021 +0800

    a quick in-test fix on compress()'s problem with bn etc.

    We add some utility functions inside `test_compress.py` to address the issue spotted in the last commit. We first alter the zeros in weight masks and bias masks to NaN. This will cause all the values dependent on to-be-pruned weight/bias to become NaN. Then, we resolve NaNs by setting NaN as zero before FC/Conv, where zero input is output-neutral. This solution is not ultimate, but any error can be spotted from NaN in model output.

* commit f5f04fe1972e6d3d961cdfa934fde559c6f6010f
Author: Jerry Xie <oceania.xie@gmail.com>
Date:   Sun Jan 10 23:28:20 2021 +0800

    reveal how SpeedUp fails to handle batchnorm bias

    We add a test script `/src/sdk/pynni/tests/test_compress.py` to show that simple zero-masking strategy of pruners will leave some channels' bias of batchnorm effective, which should have been masked as well. This problem will also apply to those operations to whom zero input is not output-neutral. This problem will finally result to different output of models after compress() and after speedup().

* commit 6f34452d407c6f449c2a0ed25d6900e76988fc55
Author: Jerry Xie <oceania.xie@gmail.com>
Date:   Fri Jan 8 12:48:10 2021 +0800

    SlimPruner is not dependency-aware, incompatible with resnet

    We add a test script `/src/sdk/pynni/tests/test_speedup_op_add.py` to reveal that SlimPruner is not compatible with `aten::add`(tensor addition), which is an essential part of residual networks. This is because SlimPruner is not dependency-aware, pruning different channels of the two input of the addition.

* commit 2522981516866d283d54707cecd8d70d56705166
Author: Jerry Xie <oceania.xie@gmail.com>
Date:   Mon Jan 4 16:42:33 2021 +0800

    fix aten::cat mask inference

    To explain: NNI infers masks for a node as `None` if the mask is a full one(i.e. no channel is going to be pruned, equivalent to `torch.arange(dim_size)`). But this fails for operation `aten::cat`(see `cat_inshape` in `/src/sdk/pynni/nni/compression/torch/speedup/infer_shape.py`) when the node has multiple predecessors but anyone(but not all) of them don't update the mask by `out_shape`. The unupdated mask, supposed to be a full mask, is taken as empty. It would not incur warning before model forwarding and would cause runtime error because the channels are not aligned.

    The fix sets the default output mask of `aten::cat` as a full mask `torch.arange(dim_size)` and prune the channels decrementally.

* commit 623d8586d2dd328f72bec52d6b3dfc227a4752bb
Author: Jerry Xie <oceania.xie@gmail.com>
Date:   Mon Dec 28 12:01:22 2020 +0800

    add mish to no_replace

    To explain: add a novel activation function Mish to supported list.

* commit 2fa2fd805d35910f748c770f86ec315ed716543d
Author: Jerry Xie <oceania.xie@gmail.com>
Date:   Mon Dec 28 11:31:27 2020 +0800

    support speed up some activations and upsample; support prim::TupleUnpack

    To explain: 1. Add some element-wise or channel-wise operations to inference list. 2. Support many-to-many operation `prim::TupleUnpack` to supported list. `prim::TupleUnpack` is a primitive used by modules to output multiple tensors. `prim::TupleUnpack` has the same number of predecessors and successors, whose masks are matched one-to-one. This fix fails when multiple successors shares one predecessor's mask.