# GPT-2 Training Notes

## Table of Contents
1. [Constraints - GPU computation and Memory bandwidth](#1-constraints---gpu-computation-and-memory-bandwidth)
2. [torch.compile](#2-torchcompile)
3. [FlashAttention instead of regular Attention](#3-flashattention-instead-of-regular-attention)
4. [Nice Numbers](#4-nice-numbers)
5. [Batch size, Gradient Accumulation and why 124M fits on a 5060 Ti](#5-batch-size-gradient-accumulation-and-why-124m-fits-on-a-5060-ti)

## Speedup Summary

| Change | Approx. speedup (vs FP32) |
|---|---|
| Default FP32 | 1 X |
| TF32 | ~1.5 X |
| BF16 (torch.autocast) | ~2 X |
| torch.compile + BF16 | ~2.5 X |
| FlashAttention | ~3 X |
| vocab_size 50304 (nice numbers) | small improvement over FlashAttention |

All runs use `train_loader = DataLoaderLite(B=8, T=512)`.

---

# How to make the code faster

## 1. Constraints - GPU computation and Memory bandwidth

### For GPU computation
We reduce the precision of the calcutations.
By default, pytorch uses float32 precision, but for Deep Learning we actually don't need it,
we can make do with less precision.

What are the other choices: FP32 (1 sign bit, 8 exponent bit, 23 mantissa bit)

| Format | Sign | Exponent | Mantissa | Notes |
|---|---|---|---|---|
| FP32 | 1 | 8 | 23 | default |
| TF32 | 1 | 8 | 10 | 8 X faster |
| FP16 | 1 | 5 | 10 | 16 X faster, reduced range |
| BF16 | 1 | 8 | 7 | |

e.g. run:
```python
train_loader = DataLoaderLite(B=8, T=512)
```

<details>
<summary><b>On Default i.e. FP32</b></summary>

```text
using device: cuda
loaded 338025 tokens
1 epoch = 82 batches
step 0, loss: 10.815738677978516, dt: 644.73ms, tok/sec: 6353.00
step 1, loss: 9.401269912719727, dt: 435.19ms, tok/sec: 9412.07
step 2, loss: 9.547967910766602, dt: 389.69ms, tok/sec: 10510.94
step 3, loss: 8.889032363891602, dt: 412.36ms, tok/sec: 9933.05
step 4, loss: 8.607047080993652, dt: 418.50ms, tok/sec: 9787.29
step 5, loss: 8.221549987792969, dt: 409.60ms, tok/sec: 9999.96
step 6, loss: 8.089510917663574, dt: 408.29ms, tok/sec: 10032.13
step 7, loss: 7.858090877532959, dt: 409.14ms, tok/sec: 10011.23
step 8, loss: 7.7790327072143555, dt: 409.50ms, tok/sec: 10002.37
step 9, loss: 7.5118937492370605, dt: 409.76ms, tok/sec: 9996.12
step 10, loss: 7.481627464294434, dt: 407.87ms, tok/sec: 10042.38
step 11, loss: 7.467465877532959, dt: 409.12ms, tok/sec: 10011.62
step 12, loss: 7.5210981369018555, dt: 409.97ms, tok/sec: 9991.05
step 13, loss: 7.43546199798584, dt: 409.21ms, tok/sec: 10009.43
step 14, loss: 7.080376625061035, dt: 410.21ms, tok/sec: 9985.11
step 15, loss: 7.0932440757751465, dt: 410.56ms, tok/sec: 9976.73
step 16, loss: 6.911331653594971, dt: 409.39ms, tok/sec: 10005.04
step 17, loss: 6.782785415649414, dt: 408.89ms, tok/sec: 10017.48
step 18, loss: 6.896854877471924, dt: 428.82ms, tok/sec: 9551.90
step 19, loss: 6.932791709899902, dt: 415.14ms, tok/sec: 9866.45
step 20, loss: 7.091119766235352, dt: 395.63ms, tok/sec: 10353.10
step 21, loss: 6.956004619598389, dt: 409.90ms, tok/sec: 9992.57
step 22, loss: 6.893294811248779, dt: 410.99ms, tok/sec: 9966.14
step 23, loss: 6.973795413970947, dt: 410.99ms, tok/sec: 9966.15
step 24, loss: 6.984163284301758, dt: 409.88ms, tok/sec: 9993.11
step 25, loss: 6.982059001922607, dt: 435.23ms, tok/sec: 9411.07
step 26, loss: 6.821687698364258, dt: 422.69ms, tok/sec: 9690.34
step 27, loss: 6.879326820373535, dt: 411.20ms, tok/sec: 9961.16
step 28, loss: 6.905203342437744, dt: 450.91ms, tok/sec: 9083.86
step 29, loss: 6.7489471435546875, dt: 416.38ms, tok/sec: 9837.06
step 30, loss: 6.7318572998046875, dt: 409.91ms, tok/sec: 9992.36
step 31, loss: 6.651180267333984, dt: 410.94ms, tok/sec: 9967.38
step 32, loss: 6.73669958114624, dt: 435.47ms, tok/sec: 9405.87
step 33, loss: 6.803301811218262, dt: 414.20ms, tok/sec: 9889.06
step 34, loss: 7.824375629425049, dt: 397.53ms, tok/sec: 10303.71
step 35, loss: 9.953006744384766, dt: 410.83ms, tok/sec: 9969.99
step 36, loss: 10.413637161254883, dt: 410.69ms, tok/sec: 9973.49
step 37, loss: 10.228466987609863, dt: 409.81ms, tok/sec: 9994.98
step 38, loss: 10.18996524810791, dt: 410.69ms, tok/sec: 9973.54
step 39, loss: 10.618829727172852, dt: 409.59ms, tok/sec: 10000.25
step 40, loss: 11.715362548828125, dt: 411.11ms, tok/sec: 9963.38
step 41, loss: 11.89215087890625, dt: 409.07ms, tok/sec: 10012.93
step 42, loss: 11.7393798828125, dt: 411.53ms, tok/sec: 9953.06
step 43, loss: 11.233799934387207, dt: 411.05ms, tok/sec: 9964.83
step 44, loss: 10.909077644348145, dt: 409.41ms, tok/sec: 10004.61
step 45, loss: 9.672836303710938, dt: 411.38ms, tok/sec: 9956.78
step 46, loss: 7.753818988800049, dt: 410.10ms, tok/sec: 9987.75
step 47, loss: 7.94093132019043, dt: 426.32ms, tok/sec: 9607.73
step 48, loss: 8.17110824584961, dt: 418.94ms, tok/sec: 9777.01
step 49, loss: 7.793941974639893, dt: 423.16ms, tok/sec: 9679.50
```
</details>

<details>
<summary><b>Using TF32: (around 1.5 X)</b></summary>

```text
using device: cuda
loaded 338025 tokens
1 epoch = 82 batches
step 0, loss: 10.933023452758789, dt: 597.10ms, tok/sec: 6859.79
step 1, loss: 9.347646713256836, dt: 313.19ms, tok/sec: 13078.26
step 2, loss: 9.08815860748291, dt: 306.55ms, tok/sec: 13361.76
step 3, loss: 8.760536193847656, dt: 301.04ms, tok/sec: 13606.09
step 4, loss: 8.429662704467773, dt: 300.22ms, tok/sec: 13643.15
step 5, loss: 7.995579242706299, dt: 301.56ms, tok/sec: 13582.49
step 6, loss: 7.859081268310547, dt: 303.77ms, tok/sec: 13483.69
step 7, loss: 7.6373677253723145, dt: 312.51ms, tok/sec: 13106.80
step 8, loss: 7.550477504730225, dt: 308.72ms, tok/sec: 13267.57
step 9, loss: 7.352997303009033, dt: 302.35ms, tok/sec: 13547.38
step 10, loss: 7.333280563354492, dt: 300.94ms, tok/sec: 13610.75
step 11, loss: 7.35683536529541, dt: 300.82ms, tok/sec: 13615.95
step 12, loss: 7.453126907348633, dt: 302.70ms, tok/sec: 13531.60
step 13, loss: 7.374917984008789, dt: 304.88ms, tok/sec: 13434.68
step 14, loss: 7.028580188751221, dt: 300.90ms, tok/sec: 13612.51
step 15, loss: 7.037542343139648, dt: 304.13ms, tok/sec: 13467.92
step 16, loss: 6.861017227172852, dt: 301.93ms, tok/sec: 13566.24
step 17, loss: 6.72517204284668, dt: 308.21ms, tok/sec: 13289.47
step 18, loss: 6.864433288574219, dt: 327.92ms, tok/sec: 12490.85
step 19, loss: 6.887475967407227, dt: 307.56ms, tok/sec: 13317.68
step 20, loss: 7.079769134521484, dt: 292.03ms, tok/sec: 14025.91
step 21, loss: 7.344615459442139, dt: 286.53ms, tok/sec: 14295.33
step 22, loss: 6.8914103507995605, dt: 305.28ms, tok/sec: 13417.33
step 23, loss: 6.971222877502441, dt: 303.00ms, tok/sec: 13518.25
step 24, loss: 7.021189212799072, dt: 301.09ms, tok/sec: 13603.77
step 25, loss: 7.001945972442627, dt: 302.68ms, tok/sec: 13532.59
step 26, loss: 6.845973968505859, dt: 301.98ms, tok/sec: 13563.73
step 27, loss: 6.875495910644531, dt: 302.10ms, tok/sec: 13558.46
step 28, loss: 6.895260810852051, dt: 301.10ms, tok/sec: 13603.28
step 29, loss: 6.757363319396973, dt: 302.47ms, tok/sec: 13541.89
step 30, loss: 6.728979110717773, dt: 301.04ms, tok/sec: 13606.06
step 31, loss: 6.644701957702637, dt: 301.93ms, tok/sec: 13566.18
step 32, loss: 6.728315830230713, dt: 302.59ms, tok/sec: 13536.53
step 33, loss: 6.780766487121582, dt: 302.13ms, tok/sec: 13557.19
step 34, loss: 6.802833557128906, dt: 300.86ms, tok/sec: 13614.42
step 35, loss: 6.802273750305176, dt: 301.60ms, tok/sec: 13580.95
step 36, loss: 6.64057731628418, dt: 302.17ms, tok/sec: 13555.44
step 37, loss: 6.7803497314453125, dt: 301.34ms, tok/sec: 13592.53
step 38, loss: 6.662315368652344, dt: 300.79ms, tok/sec: 13617.60
step 39, loss: 6.479949951171875, dt: 300.69ms, tok/sec: 13621.87
step 40, loss: 6.609460830688477, dt: 314.89ms, tok/sec: 13007.81
step 41, loss: 6.69487190246582, dt: 312.56ms, tok/sec: 13104.61
step 42, loss: 6.553706169128418, dt: 335.56ms, tok/sec: 12206.61
step 43, loss: 6.537114143371582, dt: 343.71ms, tok/sec: 11917.19
step 44, loss: 6.686448097229004, dt: 333.73ms, tok/sec: 12273.55
step 45, loss: 6.618281364440918, dt: 327.97ms, tok/sec: 12488.92
step 46, loss: 6.4942240715026855, dt: 323.26ms, tok/sec: 12671.06
step 47, loss: 6.531561851501465, dt: 310.59ms, tok/sec: 13187.81
step 48, loss: 6.519811153411865, dt: 320.95ms, tok/sec: 12761.98
step 49, loss: 6.450425624847412, dt: 321.38ms, tok/sec: 12745.16
```
</details>

> **Why not 8 X?** because even though we are doing mat_mult in TF32, all the data transfer is happening in FP32 so we are still memory bound

<details>
<summary><b>Using BF16: (approx 2 X FP32) using torch.autocast</b></summary>

```text
using device: cuda
loaded 338025 tokens
1 epoch = 82 batches
step 0, loss: 10.850982666015625, dt: 620.61ms, tok/sec: 6599.91
step 1, loss: 9.340045928955078, dt: 249.40ms, tok/sec: 16423.42
step 2, loss: 9.052215576171875, dt: 246.49ms, tok/sec: 16617.23
step 3, loss: 8.737564086914062, dt: 244.90ms, tok/sec: 16725.49
step 4, loss: 8.392387390136719, dt: 245.70ms, tok/sec: 16670.44
step 5, loss: 7.985626220703125, dt: 245.97ms, tok/sec: 16652.32
step 6, loss: 7.8470458984375, dt: 245.62ms, tok/sec: 16676.36
step 7, loss: 7.625545501708984, dt: 245.19ms, tok/sec: 16705.55
step 8, loss: 7.549095153808594, dt: 244.93ms, tok/sec: 16723.36
step 9, loss: 7.29168701171875, dt: 246.04ms, tok/sec: 16647.40
step 10, loss: 7.314182281494141, dt: 245.77ms, tok/sec: 16665.91
step 11, loss: 7.327846527099609, dt: 245.99ms, tok/sec: 16651.13
step 12, loss: 7.469993591308594, dt: 244.18ms, tok/sec: 16774.64
step 13, loss: 7.3414764404296875, dt: 245.83ms, tok/sec: 16661.67
step 14, loss: 7.0261993408203125, dt: 245.47ms, tok/sec: 16686.39
step 15, loss: 7.029762268066406, dt: 244.63ms, tok/sec: 16743.60
step 16, loss: 6.8597259521484375, dt: 246.75ms, tok/sec: 16599.52
step 17, loss: 6.699352264404297, dt: 245.26ms, tok/sec: 16700.71
step 18, loss: 6.8584747314453125, dt: 246.14ms, tok/sec: 16640.66
step 19, loss: 6.89747428894043, dt: 246.58ms, tok/sec: 16611.47
step 20, loss: 7.077949523925781, dt: 245.52ms, tok/sec: 16683.23
step 21, loss: 6.950191497802734, dt: 245.32ms, tok/sec: 16696.85
step 22, loss: 6.8760986328125, dt: 245.87ms, tok/sec: 16659.40
step 23, loss: 6.967243194580078, dt: 244.91ms, tok/sec: 16724.48
step 24, loss: 6.981533050537109, dt: 248.04ms, tok/sec: 16513.29
step 25, loss: 6.986869812011719, dt: 248.32ms, tok/sec: 16494.54
step 26, loss: 8.21670150756836, dt: 251.04ms, tok/sec: 16316.03
step 27, loss: 7.074760437011719, dt: 245.76ms, tok/sec: 16666.60
step 28, loss: 6.907585144042969, dt: 245.89ms, tok/sec: 16658.10
step 29, loss: 6.991495132446289, dt: 248.80ms, tok/sec: 16462.85
step 30, loss: 7.508684158325195, dt: 251.50ms, tok/sec: 16286.38
step 31, loss: 13.907085418701172, dt: 245.59ms, tok/sec: 16678.35
step 32, loss: 13.450506210327148, dt: 244.97ms, tok/sec: 16720.46
step 33, loss: 7.625407695770264, dt: 245.59ms, tok/sec: 16677.93
step 34, loss: 10.676281929016113, dt: 247.17ms, tok/sec: 16571.53
step 35, loss: 11.35146713256836, dt: 244.99ms, tok/sec: 16718.72
step 36, loss: 11.069365501403809, dt: 245.14ms, tok/sec: 16708.75
step 37, loss: 10.173197746276855, dt: 244.93ms, tok/sec: 16723.19
step 38, loss: 8.37109661102295, dt: 245.59ms, tok/sec: 16678.53
step 39, loss: 7.6734724044799805, dt: 246.43ms, tok/sec: 16621.57
step 40, loss: 7.550847053527832, dt: 233.82ms, tok/sec: 17517.43
step 41, loss: 7.641609191894531, dt: 245.90ms, tok/sec: 16657.30
step 42, loss: 6.813040733337402, dt: 245.32ms, tok/sec: 16696.86
step 43, loss: 7.083774089813232, dt: 244.93ms, tok/sec: 16723.14
step 44, loss: 7.138890266418457, dt: 246.09ms, tok/sec: 16644.14
step 45, loss: 7.102052688598633, dt: 246.45ms, tok/sec: 16619.88
step 46, loss: 6.924814224243164, dt: 246.29ms, tok/sec: 16630.74
step 47, loss: 6.82628059387207, dt: 245.78ms, tok/sec: 16664.99
step 48, loss: 7.19133186340332, dt: 245.11ms, tok/sec: 16710.60
step 49, loss: 7.193950653076172, dt: 246.57ms, tok/sec: 16611.96
```
</details>

---

## 2. torch.compile

- A compiler for models like gcc for c/c++
- To use: wrap the model around `torch.compule(..)`
- **Tradeoff:** now we also have compilation time but we get a faster training
- Per torch docx: speed up comes from reducing python overhead and GPU read/writes
- As the compile knows what the model will do, it removes the python interpreter,
  and make sure that we are not doing extra work
- Also, once it reads the input is read from GPU mem, it does all the required before writing in mem

> **Note:** now torch.compile uses Triton, which is not availaible for Windows, so creating a wsl-ubuntu virtual envirionmet at:
> ```bash
> source ~/venvs/gpt2/bin/activate
> ```

<details>
<summary><b>Using torch.compile with BF16: (approx 2.5 X)</b></summary>

```text
using device: cuda
loaded 338025 tokens
1 epoch = 82 batches
W1119 01:50:53.382000 498 torch/_inductor/utils.py:1731] [0/0] Not enough SMs to use max_autotune_gemm mode
step 0, loss: 10.716135025024414, dt: 20112.39ms, tok/sec: 203.66
step 1, loss: 9.44778060913086, dt: 173.99ms, tok/sec: 23541.13
step 2, loss: 10.34804916381836, dt: 163.30ms, tok/sec: 25082.59
step 3, loss: 8.855146408081055, dt: 168.47ms, tok/sec: 24312.84
step 4, loss: 9.735989570617676, dt: 169.54ms, tok/sec: 24159.67
step 5, loss: 9.787371635437012, dt: 170.89ms, tok/sec: 23967.99
step 6, loss: 9.770965576171875, dt: 169.60ms, tok/sec: 24150.64
step 7, loss: 8.725395202636719, dt: 168.82ms, tok/sec: 24262.75
step 8, loss: 8.62568473815918, dt: 169.71ms, tok/sec: 24135.20
step 9, loss: 8.190214157104492, dt: 158.44ms, tok/sec: 25852.58
step 10, loss: 7.960975646972656, dt: 156.45ms, tok/sec: 26180.28
step 11, loss: 12.13192367553711, dt: 155.58ms, tok/sec: 26327.41
step 12, loss: 8.050403594970703, dt: 161.82ms, tok/sec: 25312.31
step 13, loss: 7.997645854949951, dt: 147.67ms, tok/sec: 27737.52
step 14, loss: 7.716651916503906, dt: 168.71ms, tok/sec: 24278.25
step 15, loss: 7.657550811767578, dt: 156.07ms, tok/sec: 26245.36
step 16, loss: 7.4649152755737305, dt: 155.84ms, tok/sec: 26284.19
step 17, loss: 7.28164005279541, dt: 148.71ms, tok/sec: 27543.32
step 18, loss: 7.345306396484375, dt: 168.74ms, tok/sec: 24273.92
step 19, loss: 7.2694783210754395, dt: 165.08ms, tok/sec: 24812.27
step 20, loss: 7.423839569091797, dt: 156.83ms, tok/sec: 26118.19
step 21, loss: 7.3725714683532715, dt: 154.94ms, tok/sec: 26436.10
step 22, loss: 7.644529819488525, dt: 154.48ms, tok/sec: 26514.48
step 23, loss: 7.2227959632873535, dt: 160.85ms, tok/sec: 25464.71
step 24, loss: 7.154860019683838, dt: 169.15ms, tok/sec: 24215.21
step 25, loss: 7.1075592041015625, dt: 168.38ms, tok/sec: 24325.89
step 26, loss: 6.933401107788086, dt: 166.25ms, tok/sec: 24637.03
step 27, loss: 6.969286918640137, dt: 164.54ms, tok/sec: 24893.27
step 28, loss: 6.937955856323242, dt: 168.79ms, tok/sec: 24267.24
step 29, loss: 6.772375106811523, dt: 171.96ms, tok/sec: 23818.98
step 30, loss: 6.7531418800354, dt: 162.37ms, tok/sec: 25226.34
step 31, loss: 6.653080463409424, dt: 154.92ms, tok/sec: 26439.52
step 32, loss: 6.686824798583984, dt: 155.73ms, tok/sec: 26302.21
step 33, loss: 6.763908863067627, dt: 160.90ms, tok/sec: 25456.75
step 34, loss: 6.763100624084473, dt: 160.00ms, tok/sec: 25600.02
step 35, loss: 6.771512031555176, dt: 156.82ms, tok/sec: 26118.55
step 36, loss: 6.589890003204346, dt: 156.81ms, tok/sec: 26120.26
step 37, loss: 6.743515968322754, dt: 167.54ms, tok/sec: 24448.20
step 38, loss: 6.610710144042969, dt: 169.31ms, tok/sec: 24192.91
step 39, loss: 6.469640254974365, dt: 168.88ms, tok/sec: 24254.59
step 40, loss: 6.5746378898620605, dt: 164.20ms, tok/sec: 24944.85
step 41, loss: 6.686059951782227, dt: 168.27ms, tok/sec: 24341.16
step 42, loss: 6.529085636138916, dt: 165.45ms, tok/sec: 24757.00
step 43, loss: 6.528265953063965, dt: 171.52ms, tok/sec: 23880.56
step 44, loss: 6.670239448547363, dt: 171.92ms, tok/sec: 23825.06
step 45, loss: 6.608960151672363, dt: 169.53ms, tok/sec: 24160.45
step 46, loss: 6.490433692932129, dt: 169.46ms, tok/sec: 24170.99
step 47, loss: 6.544816017150879, dt: 161.87ms, tok/sec: 25303.96
step 48, loss: 6.532142639160156, dt: 165.49ms, tok/sec: 24751.25
step 49, loss: 6.45633602142334, dt: 157.37ms, tok/sec: 26028.57
```
</details>

---

## 3. FlashAttention instead of regular Attention

Paper: https://arxiv.org/pdf/2205.14135

> FlashAttention uses tiling to prevent materialization of the large 𝑁 × 𝑁 attention matrix
> (dotted box) on (relatively) slow GPU HBM. In the outer loop (red arrows), FlashAttention loops through
> blocks of the K and V matrices and loads them to fast on-chip SRAM. In each block, FlashAttention
> loops over blocks of Q matrix (blue arrows), loading them to SRAM, and writing the output of the attention
> computation back to HBM. Right: Speedup over the PyTorch implementation of attention on GPT-2.
> FlashAttention does not read and write the large 𝑁 × 𝑁 attention matrix to HBM, resulting in an 7.6×
> speedup on the attention computation.

### How
Never materialize the whole N X N wei matrix, and use online softmax to work as the data comes

```python
F.scaled_dot_product_attention( query, key, value, is_causal=True)
```

<details>
<summary><b>Using FlashAttention: (~3 X)</b></summary>

```text
using device: cuda
loaded 338025 tokens
1 epoch = 82 batches
W1119 02:04:27.462000 850 torch/_inductor/utils.py:1731] [0/0] Not enough SMs to use max_autotune_gemm mode
step 0, loss: 10.785750389099121, dt: 15816.01ms, tok/sec: 258.98
step 1, loss: 9.377215385437012, dt: 147.48ms, tok/sec: 27772.58
step 2, loss: 10.575733184814453, dt: 137.99ms, tok/sec: 29682.59
step 3, loss: 8.960519790649414, dt: 137.69ms, tok/sec: 29748.90
step 4, loss: 8.947587966918945, dt: 137.83ms, tok/sec: 29717.82
step 5, loss: 8.505331039428711, dt: 137.70ms, tok/sec: 29745.60
step 6, loss: 8.137837409973145, dt: 137.41ms, tok/sec: 29808.98
step 7, loss: 7.86428689956665, dt: 137.84ms, tok/sec: 29716.48
step 8, loss: 7.7730255126953125, dt: 138.09ms, tok/sec: 29661.02
step 9, loss: 7.508641242980957, dt: 137.60ms, tok/sec: 29768.38
step 10, loss: 7.4846978187561035, dt: 137.98ms, tok/sec: 29686.39
step 11, loss: 7.480250358581543, dt: 138.30ms, tok/sec: 29615.82
step 12, loss: 7.528800010681152, dt: 137.60ms, tok/sec: 29767.56
step 13, loss: 7.434748649597168, dt: 146.92ms, tok/sec: 27878.67
step 14, loss: 7.085101127624512, dt: 148.64ms, tok/sec: 27557.06
step 15, loss: 7.0594940185546875, dt: 141.34ms, tok/sec: 28979.22
step 16, loss: 6.839859962463379, dt: 137.47ms, tok/sec: 29795.02
step 17, loss: 7.768298149108887, dt: 140.30ms, tok/sec: 29193.93
step 18, loss: 6.948984622955322, dt: 150.64ms, tok/sec: 27190.01
step 19, loss: 6.9564208984375, dt: 147.62ms, tok/sec: 27747.73
step 20, loss: 7.108111381530762, dt: 147.59ms, tok/sec: 27752.35
step 21, loss: 6.9615254402160645, dt: 147.68ms, tok/sec: 27736.26
step 22, loss: 6.92085599899292, dt: 148.03ms, tok/sec: 27670.10
step 23, loss: 7.119521141052246, dt: 148.80ms, tok/sec: 27527.56
step 24, loss: 6.978789806365967, dt: 138.45ms, tok/sec: 29584.66
step 25, loss: 6.936742782592773, dt: 138.45ms, tok/sec: 29585.06
step 26, loss: 6.777704238891602, dt: 137.59ms, tok/sec: 29770.08
step 27, loss: 6.84871768951416, dt: 137.79ms, tok/sec: 29725.84
step 28, loss: 6.897599220275879, dt: 137.49ms, tok/sec: 29791.72
step 29, loss: 6.728222370147705, dt: 148.29ms, tok/sec: 27620.77
step 30, loss: 6.719959735870361, dt: 141.11ms, tok/sec: 29026.47
step 31, loss: 6.663283348083496, dt: 137.39ms, tok/sec: 29812.65
step 32, loss: 6.634895324707031, dt: 137.92ms, tok/sec: 29699.22
step 33, loss: 6.697081565856934, dt: 138.01ms, tok/sec: 29678.75
step 34, loss: 6.70925235748291, dt: 137.85ms, tok/sec: 29714.53
step 35, loss: 6.697072982788086, dt: 137.76ms, tok/sec: 29732.83
step 36, loss: 6.518827438354492, dt: 139.48ms, tok/sec: 29365.54
step 37, loss: 6.673777103424072, dt: 149.30ms, tok/sec: 27435.60
step 38, loss: 6.52894401550293, dt: 147.57ms, tok/sec: 27756.74
step 39, loss: 6.411558151245117, dt: 148.08ms, tok/sec: 27660.17
step 40, loss: 6.510305881500244, dt: 147.68ms, tok/sec: 27736.58
step 41, loss: 6.651867866516113, dt: 148.17ms, tok/sec: 27643.66
step 42, loss: 6.558408260345459, dt: 141.33ms, tok/sec: 28982.21
step 43, loss: 6.522134780883789, dt: 138.15ms, tok/sec: 29648.78
step 44, loss: 6.659873008728027, dt: 137.33ms, tok/sec: 29825.80
step 45, loss: 6.594688892364502, dt: 137.39ms, tok/sec: 29813.02
step 46, loss: 6.450817108154297, dt: 138.42ms, tok/sec: 29591.43
step 47, loss: 6.463526248931885, dt: 138.63ms, tok/sec: 29546.09
step 48, loss: 6.466276168823242, dt: 136.72ms, tok/sec: 29959.73
step 49, loss: 6.360651969909668, dt: 137.12ms, tok/sec: 29872.63
```
</details>

---

## 4. Nice Numbers

Nice numbers = having many powers of 2, like - 8, 64, 256, 1024 etc

So we just scan out code for ugly numbers i.e. prime or odd numbers and replace them with nice numbers

<details>
<summary><b>Using vocab_size of 50304: (small emprovement over FlashAttention, even though we are increasing the total computation)</b></summary>

```text
using device: cuda
loaded 338025 tokens
1 epoch = 82 batches
step 0, loss: 10.864761352539062, dt: 1726.20ms, tok/sec: 2372.84
step 1, loss: 9.366399765014648, dt: 134.19ms, tok/sec: 30524.03
step 2, loss: 10.577816009521484, dt: 132.77ms, tok/sec: 30851.27
step 3, loss: 8.815485000610352, dt: 132.55ms, tok/sec: 30900.55
step 4, loss: 8.786298751831055, dt: 132.33ms, tok/sec: 30951.82
step 5, loss: 9.481325149536133, dt: 131.81ms, tok/sec: 31075.55
step 6, loss: 8.613117218017578, dt: 132.72ms, tok/sec: 30861.63
step 7, loss: 14.117192268371582, dt: 132.88ms, tok/sec: 30825.09
step 8, loss: 8.139504432678223, dt: 132.45ms, tok/sec: 30925.58
step 9, loss: 7.738348960876465, dt: 133.15ms, tok/sec: 30762.22
step 10, loss: 7.798929691314697, dt: 132.30ms, tok/sec: 30960.35
step 11, loss: 7.7654008865356445, dt: 130.85ms, tok/sec: 31304.08
step 12, loss: 7.8330817222595215, dt: 132.75ms, tok/sec: 30854.43
step 13, loss: 7.759710788726807, dt: 133.08ms, tok/sec: 30777.87
step 14, loss: 7.384476184844971, dt: 133.65ms, tok/sec: 30647.53
step 15, loss: 7.3518171310424805, dt: 133.07ms, tok/sec: 30779.69
step 16, loss: 7.095019340515137, dt: 132.91ms, tok/sec: 30817.84
step 17, loss: 6.973043441772461, dt: 133.65ms, tok/sec: 30647.74
step 18, loss: 7.065961837768555, dt: 133.57ms, tok/sec: 30665.58
step 19, loss: 7.0203657150268555, dt: 133.78ms, tok/sec: 30616.99
step 20, loss: 7.162018775939941, dt: 133.18ms, tok/sec: 30755.45
step 21, loss: 6.96519660949707, dt: 133.38ms, tok/sec: 30708.39
step 22, loss: 6.900435447692871, dt: 134.20ms, tok/sec: 30521.48
step 23, loss: 6.978412628173828, dt: 133.55ms, tok/sec: 30670.23
step 24, loss: 6.988526821136475, dt: 133.59ms, tok/sec: 30661.09
step 25, loss: 6.968148708343506, dt: 133.15ms, tok/sec: 30762.05
step 26, loss: 6.7750349044799805, dt: 133.66ms, tok/sec: 30644.46
step 27, loss: 6.828729152679443, dt: 132.69ms, tok/sec: 30869.73
step 28, loss: 6.822756290435791, dt: 133.30ms, tok/sec: 30726.68
step 29, loss: 6.689215183258057, dt: 133.67ms, tok/sec: 30643.70
step 30, loss: 6.644038200378418, dt: 133.17ms, tok/sec: 30757.87
step 31, loss: 6.593372821807861, dt: 133.36ms, tok/sec: 30714.15
step 32, loss: 6.658700942993164, dt: 133.88ms, tok/sec: 30593.44
step 33, loss: 6.74553918838501, dt: 127.77ms, tok/sec: 32057.98
step 34, loss: 6.73042106628418, dt: 133.90ms, tok/sec: 30590.01
step 35, loss: 6.7220048904418945, dt: 133.98ms, tok/sec: 30572.43
step 36, loss: 6.536997318267822, dt: 133.85ms, tok/sec: 30602.32
step 37, loss: 7.10848331451416, dt: 132.76ms, tok/sec: 30852.66
step 38, loss: 6.599498748779297, dt: 132.84ms, tok/sec: 30834.33
step 39, loss: 6.507302761077881, dt: 133.23ms, tok/sec: 30743.17
step 40, loss: 6.657893180847168, dt: 133.45ms, tok/sec: 30692.92
step 41, loss: 6.728273391723633, dt: 133.51ms, tok/sec: 30678.39
step 42, loss: 6.571012020111084, dt: 127.19ms, tok/sec: 32203.22
step 43, loss: 6.539224624633789, dt: 133.92ms, tok/sec: 30585.49
step 44, loss: 6.679215908050537, dt: 132.98ms, tok/sec: 30802.54
step 45, loss: 6.618413925170898, dt: 133.32ms, tok/sec: 30723.00
step 46, loss: 6.507207870483398, dt: 133.26ms, tok/sec: 30736.35
step 47, loss: 6.536697864532471, dt: 133.34ms, tok/sec: 30717.89
step 48, loss: 6.537623405456543, dt: 133.24ms, tok/sec: 30741.91
step 49, loss: 6.443389892578125, dt: 133.48ms, tok/sec: 30686.06
```
</details>

---

## 5. Batch size, Gradient Accumulation and why 124M fits on a 5060 Ti

### Memory needed for training (rough estimate)

**Fixed cost (independent of batch size):**

| Item | Memory |
|---|---|
| Weights (FP32): 124M x 4 bytes | ~ 0.5 GB |
| Gradients | ~ 0.5 GB |
| AdamW state (m and v) | ~ 1 GB |
| **Total** | **~ 2 GB** → fits easily on any modern GPU |

**Activations (scale with B x T):**
- B=8, T=1024 → ~8K tokens per micro-batch, ~3 GB activations (BF16 + FlashAttention) + ~1-2 GB logits (8192 x 50304)
- B=64 → ~8 X more, ~25-30+ GB → OOM on a 16 GB 5060 Ti (Karpathy used B=64 on an 80 GB A100)
- Check actual usage with `torch.cuda.max_memory_allocated()`

So the model itself is small, it is the batch size (activations) that decides if it fits.

### What helps
- **Gradient Accumulation:** effective batch is still 524288 tokens (same as GPT-2)
  - `grad_accum_steps = 524288 / (8 * 1024) = 64` micro-steps per optimizer step
  - Mathematically equivalent to B=512, just slower
- **BF16 autocast:** halves activation memory in the matmuls
- **FlashAttention:** never stores the T x T attention matrix

### The real limit is speed, not memory
- Each optimizer step = 64 forward + backward passes
- `max_steps = 50` → ~26M tokens (fine for testing)
- Full GPT-2 run (~10B tokens, ~19K steps) would take days on a 5060 Ti vs hours on 8 x A100

### Two meanings of "batch"

**Micro-batch (`B = 8`): how much runs in PARALLEL on the GPU**
- Input is `(B, T) = (8, 1024)` → 8 independent sequences go through the model at the same time (one big matmul)
- The 1024 positions within each sequence are also parallel (causal masking)
- → 8 x 1024 = 8192 token predictions per forward pass
- Limited by GPU memory

**Effective batch (`total_batch_size = 524288` tokens): how much data goes into ONE weight update**
```
for 64 micro-steps:
    forward + backward on 8 sequences -> gradients add up in .grad
optimizer.step() -> one update using all 64 x 8 = 512 sequences
```
Micro-steps run SEQUENTIALLY, not in parallel, but the result is the same as doing 512 at once.

### Why batch size matters for training (not just speed)
The gradient from a batch is an estimate of the "true" gradient over the whole dataset:
- **Small batch** → noisy estimate, updates jump around
- **Large batch** → smoother, more reliable direction, can use a higher learning rate
- GPT-2/3 used ~0.5M tokens per update for stability

### TL;DR
- `B` = how much you compute in parallel (hardware limit)
- `total_batch_size` = how much data each update learns from (optimization choice)
- Gradient Accumulation bridges the two
