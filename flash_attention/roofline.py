import matplotlib.pyplot as plt
import numpy as np

COMPUTE_ROOF = 1.02e12   # 1.02 TFLOPS
HBM_BW       = 32e9      # 32 GB/s

# ========== Naive: ncu 数据 ==========
# FLOP: 256 tokens x 256 QK pairs x 64 dim x 2(mul+add) = 8.4 MFLOP
# 注意: 每行 QK 点积 64 FLOP, softmax 忽略, VKQ 累加 64 FLOP → 共 ~128 FLOP/对
# 256行 x 256对 x 128 FLOP ≈ 8.4 MFLOP
NAIVE_FLOP  = 256 * 256 * 128
# HBM 字节: Q(32KB) + K(32KB) + V(32KB) = 96KB 每线程?
# 实际 HBM 读 = 256 tokens x 256 tokens x 64 dim x 2B = 8.4 MB (未 cache 情况)
NAIVE_BYTES = 256 * 256 * 64 * 2
# ncu: compute=1.61%, memory=9.62%, dur=5.12ms
NAIVE_COMP_PCT = 1.61
NAIVE_MEM_PCT  = 9.62

# ========== Flash: ncu 数据 (单 head) ==========
FLASH_FLOP    = 256 * 256 * 128  # 同算法
# HBM 字节: Q(4KB 一次) + KV[64行×64dim×2B×4(K块)+4(V块)] ≈ 68KB
FLASH_BYTES  = 4*1024 + 64*64*2*8  # Q(4KB) + K(32KB)+V(32KB 分块)
# ncu: compute=3.94%, memory=25.87%, dur=4.01ms
FLASH_COMP_PCT = 3.94
FLASH_MEM_PCT  = 25.87

def ai(flop, lb):  return flop / lb
def perf_ncu(pct): return COMPUTE_ROOF * pct / 100

naive_ai   = ai(NAIVE_FLOP, NAIVE_BYTES)
flash_ai   = ai(FLASH_FLOP, FLASH_BYTES)
naive_gflops  = perf_ncu(NAIVE_COMP_PCT) / 1e9
flash_gflops  = perf_ncu(FLASH_COMP_PCT) / 1e9

print(f"Naive: AI={naive_ai:.1f} FLOP/byte,  {naive_gflops:.1f} GFLOPS (ncu compute={NAIVE_COMP_PCT}%)")
print(f"Flash: AI={flash_ai:.0f} FLOP/byte, {flash_gflops:.1f} GFLOPS (ncu compute={FLASH_COMP_PCT}%)")
print(f"  结论: Naive AI=1 → memory bound (贴 HBM 带宽墙)")
print(f"        Flash AI={flash_ai:.0f} → 跳出带宽瓶颈, occupancy 限制")

fig, ax = plt.subplots(figsize=(11, 6))

x = np.logspace(-1, 4, 200)
# 两条 roof
ax.loglog(x, np.full_like(x, COMPUTE_ROOF/1e9), 'r-',  lw=2,
          label=f'Compute Roof ({COMPUTE_ROOF/1e9:.0f} GFLOPS)')
ax.loglog(x, HBM_BW/1e9 * x,                'b-',  lw=2,
          label=f'HBM BW Roof ({HBM_BW/1e9:.0f} GB/s)')

# 标两点
ax.loglog(naive_ai, naive_gflops, 'rx', ms=14, mew=2,
          label=f'Naive  (AI={naive_ai:.1f},  {naive_gflops:.1f} GFLOPS)')
ax.loglog(flash_ai, flash_gflops, 'g^', ms=14, mew=2,
          label=f'Flash  (AI={flash_ai:.0f}, {flash_gflops:.1f} GFLOPS)')

# 箭头
ax.annotate('SRAM 消除 HBM 瓶颈', xy=(flash_ai, flash_gflops),
            xytext=(naive_ai*3, naive_gflops*3),
            arrowprops=dict(arrowstyle='->', color='darkgreen', lw=2.5),
            fontsize=11, color='darkgreen')

# 标区域
ax.text(0.5, 800, 'Memory\nBound', fontsize=11, color='blue', alpha=0.6)
ax.text(50, 800, 'Compute\nBound', fontsize=11, color='red', alpha=0.6)

ax.set_xlabel('Arithmetic Intensity (FLOP/byte)', fontsize=13)
ax.set_ylabel('Performance (GFLOPS)', fontsize=13)
ax.set_title('Roofline: Naive vs Flash Attention\n'
             'Jetson Orin Nano Super  |  sm_87  |  1.02 TFLOPS  |  32 GB/s HBM',
             fontsize=14)
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.25, which='both')
ax.set_xlim(0.3, 500)
ax.set_ylim(1, 2000)

plt.tight_layout()
plt.savefig('roofline.png', dpi=150, bbox_inches='tight')
print("\nSaved: roofline.png")
