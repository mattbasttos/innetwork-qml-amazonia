"""
qfl_innetwork.plots  —  GRÁFICOS
================================
Dois painéis separados, conforme o tipo de rede usada:

  • gerar_painel_rede_real(d, outdir)     → painel_rede_real.png (4 gráficos)
        (A) convergência, (B) acurácia final, (C) erro por nó, (F) scatter
        — retrato do SISTEMA REAL (fidelidades medidas pelo NetSquid).

  • gerar_painel_rede_uniforme(d, outdir) → painel_rede_uniforme.png (2 gráficos)
        (D) robustez, (E) heatmap
        — EXPERIMENTO CONTROLADO (todos os nós com a mesma fidelidade).

  • gerar_paineis(d, outdir) gera os dois de uma vez.
Os visuais são os mesmos de antes; só foram reorganizados em dois painéis.
Escala para qualquer número de nós.
"""

from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 120, "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.3})

BLUE, GRAY, GREEN, RED = "#1f6feb", "#888780", "#1a7f37", "#d1242f"
# paleta para N nós (cicla se houver mais nós que cores)
PAL_NOS = ["#1f6feb", "#1a7f37", "#bA7517", "#d1242f",
           "#7c4dbe", "#0098a6", "#b5179e", "#5b6b60"]
def _cor_no(k):
    return PAL_NOS[k % len(PAL_NOS)]

# compat matplotlib >= 3.9 (labels → tick_labels)
_BOXLBL = "tick_labels" if tuple(map(int, matplotlib.__version__.split(".")[:2])) >= (3, 9) else "labels"


# ── desenhos individuais (cada um recebe um eixo `ax`) ───────────────────────
def _plot_convergencia(ax, d):
    R = np.arange(1, d["rodadas"] + 1)
    for arr, cor, lab in [(d["conv_id"], GRAY, "ideal"), (d["conv_q"], BLUE, "quântico")]:
        m, sd = arr.mean(0), arr.std(0)
        ax.plot(R, m, "o-", color=cor, label=lab)
        ax.fill_between(R, m - sd, m + sd, color=cor, alpha=0.18)
    ax.set_title("(A) Convergência ± desvio (seeds)")
    ax.set_xlabel("rodada federada"); ax.set_ylabel("acurácia global"); ax.legend()

def _plot_boxplot_final(ax, d):
    bp = ax.boxplot([d["final_cent"], d["final_id"], d["final_q"]],
                    patch_artist=True, widths=0.6,
                    **{_BOXLBL: ["central", "fed-ideal", "fed-quânt"]})
    for patch, cor in zip(bp["boxes"], [RED, GRAY, BLUE]):
        patch.set_facecolor(cor); patch.set_alpha(0.4)
    ax.set_title("(B) Acurácia final — distribuição"); ax.set_ylabel("acurácia")

def _plot_erro_por_no(ax, d):
    nomes = d["nomes"]
    bp = ax.boxplot([d["erros"][n] for n in nomes], patch_artist=True, widths=0.6,
                    showfliers=False, **{_BOXLBL: nomes})
    for patch, r in zip(bp["boxes"], d["rede"]):
        patch.set_facecolor(GREEN if r["fid"] >= 0.75 else RED); patch.set_alpha(0.4)
    ax.set_title("(C) Erro de recuperação por nó"); ax.set_ylabel("erro angular (rad)")

def _plot_scatter(ax, d):
    nomes = d["nomes"]
    for k in range(len(nomes)):
        m = d["no_id"] == k
        ax.scatter(d["angs_env"][m], d["angs_rec"][m], s=8, alpha=0.5,
                   color=_cor_no(k), label=f"{nomes[k]} (F={d['rede'][k]['fid']:.2f})")
    # Limites dinâmicos: ajusta o zoom ao alcance real dos dados (com margem de 10%),
    # em vez de fixar sempre em [-π, π] — evita que os pontos fiquem amontoados no centro.
    todos = np.concatenate([d["angs_env"], d["angs_rec"]])
    lim = max(np.max(np.abs(todos)), 1e-3) * 1.1
    ax.plot([-lim, lim], [-lim, lim], "k--", alpha=0.4)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_title("(D) Ângulo enviado vs recuperado")
    ax.set_xlabel("θ enviado (rad)"); ax.set_ylabel("θ recuperado (rad)")
    ax.legend(fontsize=7)

def _plot_robustez(ax, d):
    ax.errorbar(d["rob_fid"], d["rob_mean"], yerr=d["rob_std"],
                fmt="o-", color=GREEN, capsize=4, ecolor=GRAY)
    ax.axvline(0.75, color=RED, ls="--", alpha=0.6, label="limiar")
    ax.invert_xaxis()
    ax.set_title("(E) Robustez ± desvio")
    ax.set_xlabel("fidelidade da transmissão"); ax.set_ylabel("acurácia federada"); ax.legend()

def _plot_heatmap(ax, d, fig):
    im = ax.imshow(d["H"], cmap="viridis", aspect="auto", origin="lower", vmin=0.5, vmax=0.95)
    ax.set_xticks(range(len(d["shots_h"])), labels=d["shots_h"])
    ax.set_yticks(range(len(d["fids_h"])), labels=[f"{f:.2f}" for f in d["fids_h"]])
    ax.set_xlabel("orçamento de shots"); ax.set_ylabel("fidelidade")
    ax.set_title("(F) Acurácia: fidelidade × shots")
    for i in range(len(d["fids_h"])):
        for j in range(len(d["shots_h"])):
            ax.text(j, i, f"{d['H'][i,j]:.2f}", ha="center", va="center",
                    color="white" if d["H"][i, j] < 0.8 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); ax.grid(False)


# ── PAINEL 1: REDE REAL (4 gráficos) ─────────────────────────────────────────
def gerar_painel_rede_real(d, outdir="resultados"):
    os.makedirs(outdir, exist_ok=True)
    fig, axs = plt.subplots(2, 2, figsize=(13, 9))
    _plot_convergencia(axs[0, 0], d)
    _plot_boxplot_final(axs[0, 1], d)
    _plot_erro_por_no(axs[1, 0], d)
    _plot_scatter(axs[1, 1], d)
    fig.suptitle(f"Rede real — {d['modelo']} · {d['agregador']}  (fidelidades medidas pelo NetSquid)",
                 fontsize=13, y=1.00)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "painel_rede_real.png"), bbox_inches="tight")
    plt.close(fig)


# ── PAINEL 2: REDE UNIFORME (2 gráficos) ─────────────────────────────────────
def gerar_painel_rede_uniforme(d, outdir="resultados"):
    os.makedirs(outdir, exist_ok=True)
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
    _plot_robustez(axs[0], d)
    _plot_heatmap(axs[1], d, fig)
    fig.suptitle(f"Rede uniforme (experimento controlado) — {d['modelo']} · {d['agregador']}",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "painel_rede_uniforme.png"), bbox_inches="tight")
    plt.close(fig)


def gerar_paineis(d, outdir="resultados"):
    """Gera os dois painéis (rede real + rede uniforme)."""
    gerar_painel_rede_real(d, outdir)
    gerar_painel_rede_uniforme(d, outdir)