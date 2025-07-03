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


# Nsight systems:

(url from https://developer.nvidia.com/nsight-systems/get-started)

```
wget https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2025_3/NsightSystems-linux-public-2025.3.1.90-3582212.run
chmod +x nsight-systems-*.run
sudo ./nsight-systems-*.run
```
