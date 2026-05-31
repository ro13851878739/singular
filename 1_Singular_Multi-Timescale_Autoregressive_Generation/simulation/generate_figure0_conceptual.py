"""Generate Figure 0: Singular Three-Layer Conceptual Framework Diagram."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT = os.path.dirname(os.path.abspath(__file__)) + "/../figures/fig0_conceptual_framework.png"

fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, 12)
ax.set_ylim(0, 8.5)
ax.axis("off")

# ── Colors ──
C_LAYER1 = "#1A237E"   # dark indigo — top layer
C_LAYER2 = "#283593"   # medium indigo — pillars
C_PILLAR = "#5C6BC0"   # light indigo — individual pillars
C_LAYER3 = "#E8EAF6"   # very light indigo — instantiations
C_INST1  = "#FF5722"   # deep orange — Mamba
C_INST2  = "#4CAF50"   # green — Transformer
C_INST3  = "#2196F3"   # blue — future
C_ARROW  = "#9E9E9E"
C_WHITE  = "#FFFFFF"

# ── Layer 1: Framework ──
l1 = mpatches.FancyBboxPatch((0.5, 6.3), 11, 1.5, boxstyle="round,pad=0.1",
                              facecolor=C_LAYER1, edgecolor="white", linewidth=2)
ax.add_patch(l1)
ax.text(6, 7.5, "Singular Multi-Rate Framework", ha="center", va="center",
        fontsize=18, fontweight="bold", color="white")
ax.text(6, 7.05, "Architecture-agnostic · Timescale separation via non-linear singular perturbation",
        ha="center", va="center", fontsize=10, color="#BBDEFB")
ax.text(6, 6.65, "Not every token needs the full power.",
        ha="center", va="center", fontsize=12, fontstyle="italic", color="#90CAF9")

# ── Down arrow ──
ax.annotate("", xy=(6, 6.3), xytext=(6, 5.8), arrowprops=dict(arrowstyle="->", lw=2.5, color=C_ARROW))

# ── Layer 2: Three Pillars ──
l2 = mpatches.FancyBboxPatch((0.5, 2.8), 11, 2.9, boxstyle="round,pad=0.1",
                              facecolor=C_LAYER2, edgecolor="white", linewidth=2)
ax.add_patch(l2)
ax.text(6, 5.35, "Three Enabling Pillars", ha="center", va="center",
        fontsize=14, fontweight="bold", color="white")

pillar_w, pillar_h = 3.2, 1.8
pillar_y = 3.1
pillars = [
    ("Timescale\nSeparation", 1.9, "Tikhonov's Theorem\nFast → Slow Manifold\nε = 0.04 ≪ 1"),
    ("Event-Triggered\nChange-Detection", 6.0, "Self-Supervised Gating\nΔ_semantic = 1 − cos(Ω_t, Ω_{t−3})\nModel votes with hidden state"),
    ("Input-to-State\nStability", 10.1, "Structural Contraction\nBounded Tracking Error\nZeno-Free Guarantee"),
]
for title, x, desc in pillars:
    p = mpatches.FancyBboxPatch((x - pillar_w/2, pillar_y), pillar_w, pillar_h,
                                 boxstyle="round,pad=0.05",
                                 facecolor=C_PILLAR, edgecolor="white", linewidth=2, alpha=0.9)
    ax.add_patch(p)
    ax.text(x, pillar_y + pillar_h/2 + 0.25, title, ha="center", va="center",
            fontsize=9, fontweight="bold", color="white")
    ax.text(x, pillar_y + pillar_h/2 - 0.45, desc, ha="center", va="center",
            fontsize=7, color="#E8EAF6", linespacing=1.3)

# ── Down arrows from pillars ──
for x in [1.9, 6.0, 10.1]:
    ax.annotate("", xy=(x, pillar_y), xytext=(x, 2.7), arrowprops=dict(arrowstyle="->", lw=2, color=C_ARROW))

# ── Layer 3: Instantiations ──
l3 = mpatches.FancyBboxPatch((0.5, 0.4), 11, 2.2, boxstyle="round,pad=0.1",
                              facecolor=C_LAYER3, edgecolor="#BDBDBD", linewidth=2)
ax.add_patch(l3)
ax.text(6, 2.35, "Concrete Instantiations   ·   13.95× GFLOPS Compression is a corollary, not a property of any single model",
        ha="center", va="center", fontsize=10, color="#37474F")

inst_w, inst_h = 3.2, 1.2
inst_y = 0.6
insts = [
    ("Mamba Prototype\n(this paper)", C_INST1, 1.9, "T1: mamba-2.8B (~2 Hz)\nT3: lightweight SSM (50 Hz)\nFiLM injection + ETCD"),
    ("Transformer Realization\n(future)", C_INST2, 6.0, "T1: 32–64 layer Transformer\nT3: 2–4 layer Transformer\nSlow-refreshing KV-cache"),
    ("Any Causal Architecture\n(future)", C_INST3, 10.1, "T1: heavy causal backbone\nT3: lightweight surface net\nRequires only hidden-state trajectory"),
]
for title, color, x, desc in insts:
    p = mpatches.FancyBboxPatch((x - inst_w/2, inst_y), inst_w, inst_h,
                                 boxstyle="round,pad=0.05",
                                 facecolor=color, edgecolor="white", linewidth=2, alpha=0.85)
    ax.add_patch(p)
    ax.text(x, inst_y + inst_h/2 + 0.15, title, ha="center", va="center",
            fontsize=8, fontweight="bold", color="white")
    ax.text(x, inst_y + inst_h/2 - 0.35, desc, ha="center", va="center",
            fontsize=6.5, color="white", alpha=0.9, linespacing=1.2)

plt.tight_layout()
fig.savefig(OUT, dpi=200, bbox_inches="tight")
plt.close()
print(f"✅ Saved: {OUT}")
