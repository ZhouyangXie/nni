# Notes for fixes over NNIv1.9
This is a note for the fixes over NNI of the current branch(v1.9). Trace the changes in files by 'git log'.

* commit 2522981516866d283d54707cecd8d70d56705166
Date:   Mon Jan 4 16:42:33 2021 +0800

    fix aten::cat mask inference

    To explain: NNI infers masks for a node as `None` if the mask is a full one(i.e. no channel is going to be pruned, equivalent to `torch.arange(dim_size)`). But this fails for operation `aten::cat`(see `cat_inshape` in `\src\sdk\pynni\nni\compression\torch\speedup\infer_shape.py`) when the node has multiple predecessors but anyone(but not all) of them don't update the mask by `out_shape`. The unupdated mask, supposed to be a full mask, is taken as empty. It would not incur warning before model forwarding and would cause runtime error because the channels are not aligned.

    The fix sets the default output mask of `aten::cat` as a full mask `torch.arange(dim_size)` and prune the channels decrementally.

* commit 623d8586d2dd328f72bec52d6b3dfc227a4752bb
Date:   Mon Dec 28 12:01:22 2020 +0800

    add mish to no_replace

    To explain: add a novel activation function Mish to supported list.

* commit 2fa2fd805d35910f748c770f86ec315ed716543d
Date:   Mon Dec 28 11:31:27 2020 +0800

    support speed up some activations and upsample; support prim::TupleUnpack

    To explain: 1. Add some element-wise or channel-wise operations to inference list. 2. Support many-to-many operation `prim::TupleUnpack` to supported list. `prim::TupleUnpack` is a primitive used by modules to output multiple tensors. `prim::TupleUnpack` has the same number of predecessors and successors, whose masks are matched one-to-one. This fix fails when multiple successors shares one predecessor's mask.