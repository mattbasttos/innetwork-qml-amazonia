"""
qfl_innetwork.export  —  EXPORT CSV
===================================
Exporta os dados dos experimentos em CSVs tidy (formato longo), cada um com as
colunas `modelo` e `agregador`, para você CONCATENAR resultados de diferentes
modelos/estratégias e comparar depois (em pandas, Excel, etc.).

Arquivos gerados em <outdir>/csv/:
  rede.csv                  — caracterização da rede por nó
  acuracia_final.csv        — acurácia final por seed e método
  convergencia.csv          — acurácia por rodada, seed e método
  erro_por_no.csv           — estatísticas do erro de recuperação por nó
  robustez.csv              — acurácia vs fidelidade (média ± desvio)
  heatmap.csv               — acurácia por (fidelidade, shots)
  resumo.csv                — médias finais + teste estatístico
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd


def exportar_csv(d, cfg, teste):
    outdir = os.path.join(cfg.OUTDIR, "csv")
    os.makedirs(outdir, exist_ok=True)
    M, A = d["modelo"], d["agregador"]

    def save(df, nome):
        df.insert(0, "modelo", M); df.insert(1, "agregador", A)
        path = os.path.join(outdir, nome)
        df.to_csv(path, index=False)
        return path

    # rede por nó
    save(pd.DataFrame([{k: r.get(k) for k in
            ("no", "local", "dist", "fid", "lat", "p", "shots", "perdidos")}
            for r in d["rede"]]), "rede.csv")

    # acurácia final (formato longo: seed × método)
    linhas = []
    for s in range(len(d["final_q"])):
        linhas += [
            {"seed": s, "metodo": "centralizado",   "acuracia": d["final_cent"][s]},
            {"seed": s, "metodo": "federado_ideal",  "acuracia": d["final_id"][s]},
            {"seed": s, "metodo": "federado_quantico","acuracia": d["final_q"][s]},
        ]
    save(pd.DataFrame(linhas), "acuracia_final.csv")

    # convergência (seed × rodada × método)
    linhas = []
    for met, arr in [("ideal", d["conv_id"]), ("quantico", d["conv_q"])]:
        for s in range(arr.shape[0]):
            for r in range(arr.shape[1]):
                linhas.append({"seed": s, "rodada": r + 1,
                               "metodo": met, "acuracia": float(arr[s, r])})
    save(pd.DataFrame(linhas), "convergencia.csv")

    # erro por nó (estatísticas da distribuição)
    linhas = []
    for r in d["rede"]:
        e = np.array(d["erros"][r["no"]])
        linhas.append({"no": r["no"], "fidelidade": r["fid"],
                       "erro_medio": e.mean(), "erro_std": e.std(),
                       "erro_mediana": np.median(e),
                       "erro_p25": np.percentile(e, 25),
                       "erro_p75": np.percentile(e, 75), "n": len(e)})
    save(pd.DataFrame(linhas), "erro_por_no.csv")

    # robustez
    save(pd.DataFrame({"fidelidade": d["rob_fid"],
                       "acuracia_media": d["rob_mean"],
                       "acuracia_std": d["rob_std"]}), "robustez.csv")

    # heatmap (longo)
    linhas = []
    for i, fid in enumerate(d["fids_h"]):
        for j, sh in enumerate(d["shots_h"]):
            linhas.append({"fidelidade": fid, "shots": sh,
                           "acuracia": float(d["H"][i, j])})
    save(pd.DataFrame(linhas), "heatmap.csv")

    # resumo + teste estatístico
    save(pd.DataFrame([{
        "acc_centralizado": float(np.mean(d["final_cent"])),
        "acc_fed_ideal":    float(np.mean(d["final_id"])),
        "acc_fed_quantico": float(np.mean(d["final_q"])),
        "delta_ideal_menos_quantico": teste["delta"],
        "t_pareado": teste["t"], "n_seeds": teste["n"],
    }]), "resumo.csv")

    return outdir
