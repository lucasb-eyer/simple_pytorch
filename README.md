# simple_pytorch

First step is getting simple_fsdp to work outside of torchtitan. I'm failing, pls help??

# First tests

This is with a big fat MLP, trying to understand which combination of things
have which effect. Code in speed_mem_tests_mlp.py.

### Baseline: seqlen=1024, model=1024 3xMLP

| what? | init MiB | pre-step0 MiB | peak MiB | steptime ms |
|-------|----------|---------------|----------|-------------|
| Nothing                      | 348 |  -  | 5985 | 355/340 |
| . | . | . | . | . |
| +autocast(bf16)              | 348 |  -  | 2211 | 63 |
| . | . | . | . | . |
| +data_parallel               | 348 | 432 | 5513 | 844 |
| +compile(model)              | 348 | 432 | 3347 | 1018 |
| +ac_mode="full"              | 348 | 432 | 3347 | 850 |
| +compile(step_fn)            | 348 | 432 | 2638 | 825 |
| +compile(_fwd)+compile(_bwd) | 348 | 432 | 2638 | 780 |
| +param=fp32,reduce=bf16      | 348 | 432 | 2638 | 651 |
| +param=bf16,reduce=fp32      | 348 | 432 | 1502 | 429 |
| +param=bf16,reduce=bf16      | 348 | 432 | 1502 | 294 |

### On 4xA100 SXM, seqlen=4096, model=4096 4xMLP

|          what?               | init MiB | pre-step0 MiB | peak MiB | steptime ms |
|------------------------------|----------|---------------|----------|-------------|
| Nothing                      |   3072   |      -        | 40_977   | 7020        |
| . | . | . | . | . |
| autocast (bf16)              |   3072   |      -        | 30_737   | 600         |
| compile (_fwd + _bwd)        |   3072   |      -        | 33_297   | 6978        |
| both                         |   3072   |      -        | 22_801   | 565         |
| . | . | . | . | . |
| data_parallel                |   3072   |    3200       | 36_625   | 7169        |
| +compile(2x)                 |   3072   |    3200       | 26_896   | 7141        |
| +ac_mode=full + compile(2x)  |   3072   |    3200       | 28_944   | 7118        |
| +param=bf16,reduce=bf16      |   3072   |    3200       | 15_633   | 566         |
| . | . | . | . | . |
| +compile(model, options)     |   3072   |    3200       | 8448     | 590         |
| +compile(model)              |   3072   |    3200       | 19_473   | 575         |

Let's dig into compile options:
| no option     | 19_473 | 575 |
| epifusion     | 19_473 | 569 |
| shape-padding | 19_473 | 568 |
| autotune      | 19_473 | 590 |
| cudagraph     |   8448 | 572 |
| cg+epi+pad    |   8448 | 573 |

Some weird observations:
- with `compile(model, cudagraph)` has median 8448, but actually first step 19k, second step 21k!
- with `compile(model)` has consistent 21k (after first step 19k)
- with `compile(2x)` without cudagraph, getting 15k consistently, and fastest too!

### Meta init

With 4 GPUs:
- Without: 3072 after Model(), 3200 after data_parallel(model), 15633 peak, 559ms
- With: 0 after Model() and data_parallel, 768 after to_empty, 1088 after init_weights, 15633 peak, 496ms

Seems to work, almost.


### On 2xA100 SXM, Transformer bs=8, seq=4096, dim=4096, d=4, head=128, kvred=4, grow=4

FlexAttn and FSDP and compile.

compile(fwd + bwd): 26'661MiB, 974ms
compile(model): 28'709MiB, 993ms

No effect: torch._dynamo.config.optimize_ddp = False
No effect: torch._inductor.config.reorder_for_peak_memory = False

ckpt(mlp) + compile(fwd + bwd): 17'446MiB, 1055ms
ckpt(mlp) + compile(model): 19'493MiB, 1075ms

ckpt(attn) + compile(fwd + bwd): OOM?
ckpt(attn) + compile(model): OOM?

ckpt(both) + compile(fwd + bwd): OOM?
ckpt(both) + compile(model): OOM?
ckpt(block) + compile(fwd + bwd): OOM

ckpt(attn) + compile(flex, max) + compile(fwd + bwd): 23'765MiB, 1018ms
ckpt(attn) + compile(flex) + compile(fwd + bwd): 23'765MiB, 1039ms

ckpt(both) + compile(flex) + compile(fwd + bwd): 14'550MiB, 1119ms
ckpt(both) + compile(flex) + compile(model): 16'597MiB, 11356ms
ckpt(both) + compile(flex, max) + compile(fwd + bwd): 14'550MiB, 1088ms

### Scaling behaviour

On two A100's:

| seqlen | peak MiB | step ms |
|--------|----------|---------|
| 512    |   7'827  |   142   |
| 1024   |   8'323  |   252   |
| 2048   |  10'196  |   500   |
| 4096   |  14'550  |  1119   |
| 8192   |  23'257  |  2722   |
| 16384  |  40'672  |  7631   |
| 32768  |  OOM bwd |    -    |

| depth  | peak MiB | step ms |
|--------|----------|---------|
|   2    |  10'484  |   615   |
|   4    |  14'550  |  1119   |
|   6    |  18'615  |  1600   |
|   8    |  22'681  |  2104   |
|   12   |  30'812  |  3107   |
|   16   |  38'943  |  4085   |
|   24   |  55'205  |  6053   |
|   32   |  71'467  |  8034   |

| width  | peak MiB | step ms |
|--------|----------|---------|
|   768  | badshape |    -    |
|  1024  |   5'968  |   177   |
|  2048  |   8'325  |   400   |
|  4096  |  14'550  |  1119   |
|  6144  |  23'782  |  2130   |
|  8192  |  36'118  |  3501   |
| 12288  |  67'415  |  7230   |
| 16384  |  OOM bwd |    -    |

Scaling A100's, default setting (4k, 4). **NOTE: constant PER-GPU batch-size (lbs) of 8**:

| A100's | gbs | lbs | peak MiB | step ms |
|--------|-----|-----|----------|---------|
|    1   |  8  |  8  |  20'119  |  1105   |
|    2   | 16  |  8  |  14'550  |  1119   |
|    3   | 24  |  8  |  12'705  |  1115   |
|    4   | 32  |  8  |  11'765  |  1116   |
|    6   | 48  |  8  |  10'839  |  1119   |
|    8   | 64  |  8  |  10'373  |  1114   |

Scaling A100's, default setting (4k, 4), global bs=64 (i.e. diff per-device):

| A100's | gbs | lbs  | peak MiB | step ms |
|--------|-----|------|----------|---------|
|    1   |  64 |  64  |  OOM bwd |    -    |
|    2   |  64 |  32  |  40'670  |  4352   |
|    4   |  64 |  16  |  20'472  |  2196   |
|    8   |  64 |   8  |  10'373  |  1114   |

### Skipping small ones:

This actually seems to be not worth it? Numbers here are `peak mem (MiB)/step time (ms)`.

On good interconnect it's barely any faster, even with deep model on short sequence:

```
4xA100SXM, d=8
    ReplSmall:  17882/2091
    ShardSmall: 17880/2096
4xA100SXM, d=32, s=128
    ReplSmall:  22884/548-557
    ShardSmall: 22871/559-571
```

And even on bad interconnect, it seems to be a bad idea: it's actually slower?!

```
8xRTX4090, d=4
    ReplSmall:  10375/4817-4829
    ShardSmall: 10373/4604-4612
8xRTX4090, d=32, s=128
    ReplSmall:  11672/20690-20700
    ShardSmall: 11656/18081-18087
```

I don't think it makes sense for replicated to _ever_ be slower though, so maybe I'm doing something wrong.

### MFU

With the default setting from above unless stated otherwise:

| A100's | attn | seqlen | depth | remat | gqa | MFU% |
|--------|------|--------|-------|-------|-----|------|
|   4    | flex |  4096  |   4   |  yes  |  4  | 26.0 |
|   4    | flex |  4096  |   8   |  yes  |  4  | 23.7 |
|   4    | flex |  4096  |   4   |   no  |  4  | 29.5 |
|   4    | flex |   512  |   4   |  yes  |  4  | 20.0 |
|   4    | flex |  4096  |   4   |  yes  |  1  | 24.3 |
|   .    |   .  |    .   |   .   |   .   |  .  |   .  |
|   4    | spda |  4096  |   4   |  yes  |  4  | 30.5 |
|   4    | spda |  4096  |   8   |  yes  |  4  | 28.8 |
|   4    | spda |  4096  |   4   |   no  |  4  | 35.3 |
|   4    | spda |   512  |   4   |  yes  |  4  | 20.8 |
|   4    | spda |  4096  |   4   |  yes  |  1  | 28.0 |
|   .    |   .  |    .   |   .   |   .   |  .  |   .  |
|   1    | flex |  4096  |   4   |  yes  |  4  | 26.2 |
|   2    | flex |  4096  |   4   |  yes  |  4  | 25.9 |
|   4    | flex |  4096  |   4   |  yes  |  4  | 26.0 |

Interestingly, using `mode=replicate` leads to lower MFU:
30.4 for spda, 25.8 for flex,
so that's something suspicious to look into at some point.


# TODOs:

- Multi-node speedtest.
- Save/load model.
- Why is `mode=replicate` slightly slower?
- check out this warning when using d=12 or more:

```
[rank1]:W0706 09:22:12.068000 17904 torch/_dynamo/convert_frame.py:1047] [5/8] torch._dynamo hit config.recompile_limit (8)
[rank1]:W0706 09:22:12.068000 17904 torch/_dynamo/convert_frame.py:1047] [5/8]    function: 'forward' (/root/main.py:100)
[rank1]:W0706 09:22:12.068000 17904 torch/_dynamo/convert_frame.py:1047] [5/8]    last reason: 5/7: ___check_type_id(self, 360762944)
[rank1]:W0706 09:22:12.068000 17904 torch/_dynamo/convert_frame.py:1047] [5/8] To log all recompilation reasons, use TORCH_LOGS="recompiles".
[rank1]:W0706 09:22:12.068000 17904 torch/_dynamo/convert_frame.py:1047] [5/8] To diagnose recompilation issues, see https://pytorch.org/docs/main/torch.compiler_troubleshooting.html.
[rank0]:W0706 09:22:12.122000 17903 torch/_dynamo/convert_frame.py:1047] [5/8] torch._dynamo hit config.recompile_limit (8)
[rank0]:W0706 09:22:12.122000 17903 torch/_dynamo/convert_frame.py:1047] [5/8]    function: 'forward' (/root/main.py:100)
[rank0]:W0706 09:22:12.122000 17903 torch/_dynamo/convert_frame.py:1047] [5/8]    last reason: 5/7: ___check_type_id(self, 633153152)
[rank0]:W0706 09:22:12.122000 17903 torch/_dynamo/convert_frame.py:1047] [5/8] To log all recompilation reasons, use TORCH_LOGS="recompiles".
[rank0]:W0706 09:22:12.122000 17903 torch/_dynamo/convert_frame.py:1047] [5/8] To diagnose recompilation issues, see https://pytorch.org/docs/main/torch.compiler_troubleshooting.html.
```

See o3: https://chatgpt.com/c/686a407e-9858-8001-acfa-1256b51107c7


# Nsight systems:

(url from https://developer.nvidia.com/nsight-systems/get-started)

```
wget https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2025_3/NsightSystems-linux-public-2025.3.1.90-3582212.run
chmod +x nsight-systems-*.run
sudo ./nsight-systems-*.run
```
