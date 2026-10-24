"""Plot Figures 2 and 3 from the output of ``diagnostics.py summarize``.

    loopmup-plot --summary summary.json --out figures

writes ``fig2_forward_norm.pdf``, ``fig3_increment_cosine.pdf``, and
``fig3_one_step_change.pdf``. It prints the fitted log-log slopes (K >= 2)
of the Figure 2 curves and the off-diagonal statistics of the
cosine matrices in Figure 3.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COLORS = {"deep": "#384CBE", "loop": "#DC702A"}
FAMILIES = [("deep", "Untied"), ("loop", "Looped")]
ALPHAS = [(0.0, "0"), (0.5, "1/2"), (1.0, "1"), (1.5, "3/2")]


def log2_axis(ax, ks) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks, [str(k) for k in ks])
    ax.minorticks_off()


def slope(ks, ys, k_min=2) -> float:
    ks, ys = np.asarray(ks, float), np.asarray(ys, float)
    m = ks >= k_min
    return float(np.polyfit(np.log(ks[m]), np.log(ys[m]), 1)[0])


def forward_norm(settings, out: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.6), sharey=True)
    for ax, (alpha, frac) in zip(axes, ALPHAS):
        for family, label in FAMILIES:
            init = settings[f"{family}_a{alpha:g}"]["init"]
            ks = [p["k"] for p in init]
            ys = [p["mean_energy"] ** 0.5 for p in init]
            ax.plot(ks, ys, "o-", color=COLORS[family], ms=3, label=label)
            print(f"Figure 2, multiplier K^-{alpha:g}: {label} norm slope "
                  f"{slope(ks, ys):+.2f}")
        log2_axis(ax, ks)
        ax.set_yscale("log")
        ax.set_title("multiplier 1" if alpha == 0 else
                     rf"$N^{{-{frac}}}$ or $m_L^{{-{frac}}}$", fontsize=9)
        ax.set_xlabel(r"loop count $N$ or depth multiplier $m_L$", fontsize=8)
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel(r"$\|x\|_{\mathrm{rms}}$ at initialization")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig2_forward_norm.pdf")


def increment_cosine(settings, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    for ax, (family, label) in zip(axes, FAMILIES):
        cos = settings[f"{family}_a0"]["increment_cosine"]
        m = np.asarray(cos["mean"])
        off = m[~np.eye(len(m), dtype=bool)]
        print(f"Figure 3 (left), {label}, K={cos['k']}: off-diagonal "
              f"cosine mean {off.mean():.3f}, min {off.min():.3f}")
        im = ax.imshow(m, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_title(f"{label}, K={cos['k']}", color=COLORS[family])
        ax.set_xlabel("increment index j")
    axes[0].set_ylabel("increment index i")
    fig.colorbar(im, ax=axes, label="cosine similarity")
    fig.savefig(out / "fig3_increment_cosine.pdf")


def one_step_change(settings, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(4, 3))
    for alpha, frac in ALPHAS:
        pts = [p for p in settings[f"loop_a{alpha:g}"]["step_change"]
               if p["step"] == 1]
        ks = [p["k"] for p in pts]
        ys = [p["mean_rms"] for p in pts]
        err = [p["sem_rms"] for p in pts]
        label = rf"$\alpha_N={frac}$" + (" (Loop-μP)" if alpha == 1 else "")
        ax.errorbar(ks, ys, yerr=err, marker="o", ms=3, capsize=2, label=label)
    log2_axis(ax, ks)
    ax.set_yscale("log")
    ax.set_xlabel(r"loop count $N$")
    ax.set_ylabel("RMS change of $x^N$ after one step")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out / "fig3_one_step_change.pdf")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--summary", default="summary.json")
    p.add_argument("--out", default="figures")
    args = p.parse_args()
    settings = json.loads(Path(args.summary).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    forward_norm(settings, out)
    increment_cosine(settings, out)
    one_step_change(settings, out)
    print(f"wrote figures to {out}/")


if __name__ == "__main__":
    main()
