"""
qfl_innetwork.data  —  DADOS
============================
Geração de dados ambientais sintéticos (prova de conceito).
Troque `gerar_particao` por um carregador de dataset real quando quiser
(mantendo a saída: X escalado para [0, π] e y em {-1, +1}).
"""

from __future__ import annotations
import numpy as np


def gerar_particao(n, seed, n_features=4):
    """
    4 features (temp, PM2.5, umidade do solo, NDVI) → rótulo de risco {-1,+1}.
    Fronteira suave (sigmoide + interação) para o ruído de rede degradar de forma
    realista. X escalado para [0, π] (pronto para AngleEmbedding).
    """
    rng = np.random.default_rng(seed)
    Xp = np.column_stack([rng.uniform(20, 40, n),     # temperatura
                          rng.uniform(5, 160, n),     # PM2.5
                          rng.uniform(0.05, 0.6, n),  # umidade do solo
                          rng.uniform(0.1, 0.9, n)])  # NDVI
    z = (0.045 * (Xp[:, 0] - 30) + 0.020 * (Xp[:, 1] - 60)
         - 3.0 * (Xp[:, 2] - 0.3) - 2.4 * (Xp[:, 3] - 0.5)
         + 0.0013 * (Xp[:, 0] - 30) * (Xp[:, 1] - 60))
    r = 1.0 / (1.0 + np.exp(-z)) + rng.normal(0, 0.04, n)
    y = np.where(r > 0.5, 1.0, -1.0)
    X = (Xp - Xp.min(0)) / (Xp.max(0) - Xp.min(0)) * np.pi
    return X, y
