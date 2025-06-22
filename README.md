# simple_pytorch

First step is getting simple_fsdp to work outside of torchtitan. I'm failing, pls help??

Running:

```
env CUDA_VISIBLE_DEVICES=1 torchrun --nproc-per-node=gpu main.py
```

Gives:

```
['cudagraphs', 'inductor', 'onnxrt']
First linear weights:  tensor([ 2.4542e-02,  2.5251e-02, -1.9412e-03,  ..., -6.2052e-05,
        -1.4805e-02, -2.7190e-02], device='cuda:0',
       grad_fn=<_ToTorchTensorBackward>)
[rank0]: Traceback (most recent call last):
[rank0]:   File "/root/main.py", line 115, in <module>
[rank0]:     main()
[rank0]:   File "/root/main.py", line 94, in main
[rank0]:     logits = model(data)
[rank0]:              ^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py", line 1773, in _wrapped_call_impl
[rank0]:     return self._call_impl(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py", line 1784, in _call_impl
[rank0]:     return forward_call(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/root/main.py", line 41, in forward
[rank0]:     x = blk(x)
[rank0]:         ^^^^^^
[rank0]:   File "/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py", line 1773, in _wrapped_call_impl
[rank0]:     return self._call_impl(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py", line 1784, in _call_impl
[rank0]:     return forward_call(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/root/main.py", line 24, in forward
[rank0]:     y = self.l1(x)
[rank0]:         ^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py", line 1773, in _wrapped_call_impl
[rank0]:     return self._call_impl(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py", line 1784, in _call_impl
[rank0]:     return forward_call(*args, **kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.11/dist-packages/torch/nn/modules/linear.py", line 125, in forward
[rank0]:     return F.linear(input, self.weight, self.bias)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]: RuntimeError: mat2 must be a matrix, got 1-D tensor
```

Note that the weights of the first linear lost a dimension!? The same happens with `replicate` btw.
