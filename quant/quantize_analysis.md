# INT8 / Q4 量化精度分析

## 测试环境

- 硬件：Jetson Orin Nano Super 8GB (CUDA 12.6, sm_87)
- 框架：llama.cpp (CUDA 后端)
- 测试文本：中文技术描述文本，6282 bytes

## 结果汇总

| 模型 | 大小 | 有效bit | tok/s | PPL | 速度 vs Q8_0 | 精度损失 |
|------|------|--------|-------|-----|-------------|---------|
| Q8_0 | 507M | ~8.5 | 74.2 | 1.0282 | baseline | baseline |
| Q4_K_M | 380M | ~4.5 | 77.37 | 1.0317 | +4.3% | **+0.34%** |
| Q4_0 | 336M | ~4.5 | 83.18 | 1.0448 | +12.1% | +1.61% |

## 分析

### Q4_0 — 最快但精度损失最大

每 32 个权重共享一把 16-bit 尺子，所有层一律 4-bit。

- 速度提升 12.1% 来自模型从 507M → 336M，每个 token 读取数据量减少 34%
- PPL 升高 1.61%，说明 naive 量化对少数敏感层造成较大误差
- 适合：显存极度受限、精度要求不高的场景

### Q4_K_M — 性价比最优

k-quant 使用 importance matrix 校准，对重要权重分配更多 bit。

- 速度提升 4.3%，模型从 507M → 380M（比 Q4_0 大 44MB）
- PPL 仅升高 0.34%，精度损失只有 Q4_0 的 **五分之一**
- 多花的 44MB 全在 attention 层和 embedding 层上
- 适合：端侧部署的默认选择，面试最值得讲

### 实际速度收益解析

decode 阶段是 memory bound 的：
- 每次生成一个 token，需要读取全部模型权重
- Q8_0: 507MB × 74.2 tok/s ≈ 37.6 GB/s 有效带宽
- Q4_K_M: 380MB × 77.4 tok/s ≈ 29.4 GB/s 有效带宽
- Q4_0: 336MB × 83.2 tok/s ≈ 27.9 GB/s 有效带宽

模型缩小后有效带宽需求降低，说明有额外的计算开销（dequant kernel）。

## 面试要点

- **为什么选 Q4_K_M 而非 Q4_0**：k-quant 用 importance matrix 校准，精度损失只有 0.34% vs 1.61%
- **量化误差来源**：每个 block（32 个权重）共享 scale，block 内 outlier 权重被截断
- **per-channel vs per-tensor**：k-quant 在不同层分配不同位宽，本质上比 per-tensor 精细
- **dp4a 指令**：Orin Ampere 支持 INT8 dp4a，量化 kernel 可硬件加速
