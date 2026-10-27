"""
qfl_innetwork.config  —  SETUP
==============================
Toda a configuração do experimento num único lugar (dataclass `Config`).
Para testar variações, basta criar um novo Config:

    cfg = Config(N_QUBITS=6, N_LAYERS=6, MODELO="VQC_6q")

ou alterar a topologia, o número de seeds, etc.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import pennylane as qml


@dataclass
class Config:
    # ── Modelo QML ──
    N_QUBITS:   int = 4        # nº de qubits = nº de features
    N_LAYERS:   int = 4        # profundidade do circuito variacional
    LR:         float = 0.15   # taxa de aprendizado do Adam
    EPOCAS_LOC: int = 6        # épocas de treino local por rodada

    # ── Federado ──
    RODADAS:    int = 8        # rodadas federadas

    # ── Rede quântica ──
    BASE_SHOTS: int = 256      # orçamento de medições do melhor enlace
    N_RUNS_NET: int = 150       # repetições Monte Carlo por enlace (NetSquid)

    # ── Análise estatística ──
    N_SEEDS:    int = 30       # repetições do experimento
    N_LOCAL:    int = 100       # amostras por nó
    N_TESTE:    int = 300      # amostras do teste global
    SEED:       int = 7        # semente base

    # ── Identificação e saída ──
    MODELO:     str = "VQC_StronglyEntangling"   # nome estampado nos CSVs
    AGREGADOR:  str = "FedAvg"                    # nome estampado nos CSVs
    OUTDIR:     str = "resultados"

    # ── Topologia: (rótulo, localidade, distância_km) ──
    TOPOLOGIA: list = field(default_factory=lambda: [
        ("S1", "Belém (hub)",        5.0),
        ("S2", "Marajó",            25.0),
        ("S3", "Tapajós",           60.0),
        ("S4", "Floresta profunda", 120.0),
    ])

    @property
    def shape_pesos(self):
        """Formato do tensor de pesos do StronglyEntanglingLayers."""
        return qml.StronglyEntanglingLayers.shape(self.N_LAYERS, self.N_QUBITS)
