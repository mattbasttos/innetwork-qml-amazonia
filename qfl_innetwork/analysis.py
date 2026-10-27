"""
qfl_innetwork.analysis  —  TESTE ESTATÍSTICO
============================================
Teste t pareado entre a acurácia final ideal e a quântica (entre seeds).
"""

from __future__ import annotations
import numpy as np


def teste_estatistico(d):
    qi = np.array(d["final_q"]); idl = np.array(d["final_id"])
    dif = idl - qi
    n = len(dif)
    media = float(np.mean(dif))
    sd = float(np.std(dif, ddof=1)) if n > 1 else 0.0
    t = media / (sd / np.sqrt(n)) if (sd > 0 and n > 1) else 0.0
    return {"delta": media, "t": t, "n": n,
            "acc_cent": float(np.mean(d["final_cent"])),
            "acc_ideal": float(np.mean(d["final_id"])),
            "acc_quant": float(np.mean(d["final_q"]))}
