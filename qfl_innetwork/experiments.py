"""
qfl_innetwork.experiments  —  EXPERIMENTOS
==========================================
Roda o experimento federado várias vezes (seeds) e coleta toda a telemetria
necessária para os gráficos e os CSVs. Recebe os componentes PLUGÁVEIS
(modelo, transmissão quântica, agregador) — então trocar qualquer um deles
muda o experimento sem alterar este arquivo.
"""

from __future__ import annotations
import time
import numpy as np

from . import data as data_mod
from . import federated as fed
from . import netsquid_backend as net


def coletar(cfg, model, aggregator, transmissao_quantica, rede, verbose=True):
    """
    Retorna um dicionário com tudo que plots.py e export.py precisam.
    `rede` traz as fidelidades reais medidas pelo NetSquid.
    """
    log = print if verbose else (lambda *a, **k: None)
    nomes = [r["no"] for r in rede]
    n_nos = len(rede)                       # nº de nós (escala automática)
    ideal = fed.IdealTransmission()

    # ── Bloco 1: convergência, acurácia final, erro por nó (N_SEEDS) ──
    log(f"  coletando — {cfg.N_SEEDS} seeds × {cfg.RODADAS} rodadas")
    conv_q, conv_id = [], []
    final_cent, final_id, final_q = [], [], []
    erros_acc = {n: [] for n in nomes}
    t0 = time.time()
    for s in range(cfg.N_SEEDS):
        parts = [data_mod.gerar_particao(cfg.N_LOCAL, 100 + s * 10 + i)
                 for i in range(len(rede))]
        Xte, yte = data_mod.gerar_particao(cfg.N_TESTE, 9000 + s)
        w0 = model.init_weights(1000 + s)

        h_id = fed.run_federated(model, rede, parts, Xte, yte, w0, ideal,
                                 aggregator, cfg.RODADAS, cfg.EPOCAS_LOC, cfg.LR)
        h_q, er = fed.run_federated(model, rede, parts, Xte, yte, w0,
                                    transmissao_quantica, aggregator,
                                    cfg.RODADAS, cfg.EPOCAS_LOC, cfg.LR,
                                    coletar_erros=True)
        c = fed.treino_centralizado(model, parts, Xte, yte, w0,
                                    cfg.RODADAS * cfg.EPOCAS_LOC, cfg.LR)
        conv_id.append(h_id); conv_q.append(h_q)
        final_id.append(h_id[-1]); final_q.append(h_q[-1]); final_cent.append(c)
        for n in nomes:
            erros_acc[n].extend(er[n])
        log(f"    seed {s+1}/{cfg.N_SEEDS}  central={c:.3f}  ideal={h_id[-1]:.3f}  quant={h_q[-1]:.3f}")
    log(f"  bloco 1: {time.time()-t0:.1f}s")

    # ── Bloco 2: robustez (varre fidelidade, várias seeds) ──
    cfgs = [(1.00, 0), (0.95, 256), (0.90, 128), (0.85, 80),
            (0.80, 48), (0.75, 32), (0.70, 20), (0.62, 12)]
    rob_fid, rob_mean, rob_std = [], [], []
    for fid, shots in cfgs:
        accs = []
        for s in range(max(2, cfg.N_SEEDS // 2)):
            parts = [data_mod.gerar_particao(cfg.N_LOCAL, 100 + s * 10 + i) for i in range(n_nos)]
            Xte, yte = data_mod.gerar_particao(cfg.N_TESTE, 9000 + s)
            w0 = model.init_weights(1000 + s)
            tx = ideal if not shots else transmissao_quantica
            h = fed.run_federated(model, net.rede_uniforme(fid, shots, n_nos), parts, Xte, yte,
                                  w0, tx, aggregator, cfg.RODADAS, cfg.EPOCAS_LOC, cfg.LR)
            accs.append(h[-1])
        rob_fid.append(fid); rob_mean.append(float(np.mean(accs))); rob_std.append(float(np.std(accs)))
    log("  robustez ok")

    # ── Bloco 3: heatmap (fidelidade × shots) ──
    fids_h  = [0.95, 0.85, 0.75, 0.65, 0.55]
    shots_h = [16, 32, 64, 128]
    H = np.zeros((len(fids_h), len(shots_h)))
    parts = [data_mod.gerar_particao(cfg.N_LOCAL, 100 + i) for i in range(n_nos)]
    Xte, yte = data_mod.gerar_particao(cfg.N_TESTE, 9000)
    w0 = model.init_weights(1000)
    for i, fid in enumerate(fids_h):
        for j, sh in enumerate(shots_h):
            h = fed.run_federated(model, net.rede_uniforme(fid, sh, n_nos), parts, Xte, yte,
                                  w0, transmissao_quantica, aggregator,
                                  4, cfg.EPOCAS_LOC, cfg.LR)
            H[i, j] = h[-1]
    log("  heatmap ok")

    # ── Bloco 4: ângulos enviados vs recuperados (1 transmissão por nó) ──
    angs_env, angs_rec, no_id = [], [], []
    parts = [data_mod.gerar_particao(cfg.N_LOCAL, 100 + i) for i in range(n_nos)]
    w0 = model.init_weights(7)
    for k, r in enumerate(rede):
        wl, _ = model.train_local(w0, parts[k][0], parts[k][1], cfg.EPOCAS_LOC, cfg.LR)
        rx = transmissao_quantica.transmit(wl, r)
        a = np.arctan2(np.sin(wl.ravel()), np.cos(wl.ravel()))
        b = np.arctan2(np.sin(rx.ravel()), np.cos(rx.ravel()))
        angs_env.extend(a.tolist()); angs_rec.extend(b.tolist())
        no_id.extend([k] * len(a))

    return dict(conv_id=np.array(conv_id), conv_q=np.array(conv_q),
                final_cent=final_cent, final_id=final_id, final_q=final_q,
                erros=erros_acc, rede=rede, nomes=nomes,
                rob_fid=rob_fid, rob_mean=rob_mean, rob_std=rob_std,
                H=H, fids_h=fids_h, shots_h=shots_h,
                angs_env=np.array(angs_env), angs_rec=np.array(angs_rec),
                no_id=np.array(no_id),
                rodadas=cfg.RODADAS, modelo=cfg.MODELO, agregador=aggregator.nome)
