#!/usr/bin/env python3
"""
main.py  —  ORQUESTRADOR
========================
Monta os componentes plugáveis e roda o pipeline completo:
  rede (NetSquid) → experimentos → teste estatístico → CSV → gráficos.

TROCAR DE MODELO ou ESTRATÉGIA (exemplos):
  from qfl_innetwork import VQCModel, QNNModel, FedAvg, FedMedian
  model = QNNModel(cfg.N_QUBITS, cfg.N_LAYERS)      # outro modelo de treino
  aggregator = FedMedian()                          # outra estratégia federada
  cfg = Config(MODELO="QNN", AGREGADOR="FedMedian") # estampar nos CSVs

Execução:
  python main.py
"""

from qfl_innetwork.config import Config
from qfl_innetwork.qml import VQCModel
from qfl_innetwork.federated import QuantumStateTransmission, FedAvg
from qfl_innetwork import netsquid_backend as net
from qfl_innetwork import experiments, analysis, export, plots

SEP = "═" * 72


def main():
    cfg = Config()                          # ← ajuste a configuração aqui
    print(SEP); print("  QFL IN-NETWORK (modular) — NetSquid + PennyLane"); print(SEP)

    # ── componentes plugáveis ──
    model       = VQCModel(cfg.N_QUBITS, cfg.N_LAYERS)   # troque por QNNModel(...)
    aggregator  = FedAvg()                               # troque por FedMedian()
    transmissao = QuantumStateTransmission()
    cfg.MODELO, cfg.AGREGADOR = type(model).__name__, aggregator.nome

    # PASSO 1 — rede quântica real (NetSquid)
    print("\n[1] Medindo a rede quântica (NetSquid)...")
    rede = net.levantar_rede(cfg)
    for r in rede:
        print(f"    {r['no']} {r['local']}: F={r['fid']:.3f}  "
              f"lat={r['lat']*1e3:.3f}ms  p={r['p']:.3f}  shots={r['shots']}")

    # PASSO 2 — experimentos
    print("\n[2] Rodando experimentos...")
    d = experiments.coletar(cfg, model, aggregator, transmissao, rede)

    # PASSO 3 — teste estatístico
    teste = analysis.teste_estatistico(d)
    print(f"\n[3] Teste: Δ(ideal−quântico)={teste['delta']:+.4f}  t={teste['t']:+.3f}  n={teste['n']}")
    print(f"    central={teste['acc_cent']:.4f}  ideal={teste['acc_ideal']:.4f}  quant={teste['acc_quant']:.4f}")

    # PASSO 4 — CSV
    pasta_csv = export.exportar_csv(d, cfg, teste)
    print(f"\n[4] CSVs exportados em: {pasta_csv}/")

    # PASSO 5 — gráficos
    plots.gerar_graficos(d, cfg.OUTDIR)
    print(f"[5] Gráficos salvos em: {cfg.OUTDIR}/")
    print("\n" + SEP)
    print(f"  Modelo: {cfg.MODELO} | Agregador: {cfg.AGREGADOR} | concluído.")
    print(SEP + "\n")


if __name__ == "__main__":
    main()
