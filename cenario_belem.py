#!/usr/bin/env python3
"""
cenario_belem.py  —  CENÁRIO: Belém e Região Metropolitana
==========================================================
Topologia da rede quântica na escala metropolitana de Belém/PA, com o hub no
GERCOM/UFPA. Distâncias curtas (≤ 17 km) — escala realista de redes quânticas.

Nós:
  GERCOM/UFPA (Hub) ...... 0,2 km
  PCT Guamá .............. 1,5 km
  Museu Emílio Goeldi .... 3,0 km
  SEMAS .................. 5,0 km
  SEMA (Ananindeua) ..... 17,0 km

Saídas:
  resultados_belem/painel_rede_real.png       (4 gráficos — rede real)
  resultados_belem/painel_rede_uniforme.png   (2 gráficos — rede uniforme)
  resultados_belem/csv/*.csv                  (dados para comparação)

Execução:
  python cenario_belem.py
"""

from qfl_innetwork.config import Config
from qfl_innetwork.qml import VQCModel
from qfl_innetwork.federated import QuantumStateTransmission, FedAvg
from qfl_innetwork import netsquid_backend as net
from qfl_innetwork import experiments, analysis, export, plots

SEP = "═" * 72

# ── Topologia de Belém e Região Metropolitana (rótulo, local, distância_km) ──
TOPOLOGIA_BELEM = [
    ("GERCOM", "GERCOM/UFPA (Hub)",        0.2),
    ("PCT",    "PCT Guamá",                1.5),
    ("MPEG",   "Museu Emílio Goeldi",      3.0),
    ("SEMAS",  "SEMAS",                    5.0),
    ("SEMA",   "SEMA (Ananindeua)",       17.0),
]


def main():
    print(SEP); print("  CENÁRIO BELÉM — QFL in-network (NetSquid + PennyLane)"); print(SEP)

    # Configuração do cenário (dados sintéticos, por enquanto)
    cfg = Config(TOPOLOGIA=TOPOLOGIA_BELEM, OUTDIR="resultados_belem")

    # Componentes plugáveis
    model       = VQCModel(cfg.N_QUBITS, cfg.N_LAYERS)
    aggregator  = FedAvg()
    transmissao = QuantumStateTransmission()
    cfg.MODELO, cfg.AGREGADOR = type(model).__name__, aggregator.nome

    # PASSO 1 — mede a rede quântica real (NetSquid)
    print("\n[1] Medindo a rede quântica de Belém (NetSquid)...")
    rede = net.levantar_rede(cfg)
    for r in rede:
        print(f"    {r['no']:7s} {r['local']:24s} d={r['dist']:5.1f}km  "
              f"F={r['fid']:.3f}  lat={r['lat']*1e3:.3f}ms  p={r['p']:.3f}  shots={r['shots']}")

    # PASSO 2 — experimentos (rede real + varreduras uniformes)
    print("\n[2] Rodando experimentos...")
    d = experiments.coletar(cfg, model, aggregator, transmissao, rede)

    # PASSO 3 — teste estatístico
    teste = analysis.teste_estatistico(d)
    print(f"\n[3] Δ(ideal−quântico)={teste['delta']:+.4f}  t={teste['t']:+.3f}  n={teste['n']}")
    print(f"    central={teste['acc_cent']:.4f}  ideal={teste['acc_ideal']:.4f}  quant={teste['acc_quant']:.4f}")

    # PASSO 4 — CSV
    pasta = export.exportar_csv(d, cfg, teste)
    print(f"\n[4] CSVs em: {pasta}/")

    # PASSO 5 — os dois painéis
    plots.gerar_paineis(d, cfg.OUTDIR)
    print(f"[5] Painéis em: {cfg.OUTDIR}/")
    print("      • painel_rede_real.png      (A convergência, B acurácia, C erro/nó, D scatter)")
    print("      • painel_rede_uniforme.png  (E robustez, F heatmap)")
    print("\n" + SEP + "\n")


if __name__ == "__main__":
    main()
