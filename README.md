# Flash Attention CUDA from Scratch

在 Jetson Orin Nano Super 上从零手写 CUDA Flash Attention kernel，含两个版本：

## 文件

| 文件 | 说明 |
|------|------|
| `flash_attn_mini.cu` | 单头 Flash Attention，含 CPU 参考 + 朴素 kernel + tiled kernel |
| `flash_attn_mini_s1.cu` | 升级版：多头 GQA (14/2) + 变长 KV cache |

## 特性

- **Tiling**: Q 常驻 SRAM，K/V 分块加载，Br=32, Bc=64
- **Online softmax**: 分块重新缩放旧累加器
- **GQA**: 14 个 Q 头共享 2 个 KV 头
- **KV cache**: 支持变长 kv_max
- **half2**: FP16 半精度，指令级并行

## 编译运行

```bash
nvcc -arch=sm_87 -O3 flash_attn_mini_s1.cu -o flash_attn_mini_s1
./flash_attn_mini_s1
```

## 性能 (Jetson Orin Nano Super, sm_87)

| Kernel | 误差 vs CPU |
|--------|------------|
| Naive | 0.000015 |
| Flash Tiled | 0.000015 |

## 扩展

[paged-flash-attention-cuda](https://github.com/456564/paged-flash-attention-cuda) — 在此 kernel 基础上加入 block_table 实现 PagedAttention。
