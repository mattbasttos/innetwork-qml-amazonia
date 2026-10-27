"""
qfl_innetwork.netsquid_backend  —  NETSQUID (rede quântica)
==========================================================
Simula a rede quântica e mede (fidelidade, latência) de cada enlace.
NetSquid é OBRIGATÓRIO; o programa encerra com mensagem clara se faltar.

Para trocar o backend de rede (ex.: outro simulador, ou um modelo analítico),
implemente uma função com a mesma assinatura de `simular_link_netsquid` e
ajuste `levantar_rede`.
"""

from __future__ import annotations

# NetSquid é obrigatório; importação protegida só para dar mensagem amigável.
try:
    import netsquid as ns
    from netsquid.components import QuantumChannel
    from netsquid.components.models import (
        FibreDelayModel, FibreLossModel, DepolarNoiseModel,
    )
    from netsquid.qubits import qubitapi as qapi
    from netsquid.qubits import ketstates as ks
    NETSQUID_OK = True
except Exception as _e:
    NETSQUID_OK = False
    _ERRO = str(_e)


def exigir_netsquid():
    """Encerra o programa se o NetSquid não estiver instalado."""
    if not NETSQUID_OK:
        print("\n" + "═" * 72)
        print("  ERRO: este projeto EXIGE o NetSquid.")
        print("  pip install --extra-index-url https://pypi.netsquid.org netsquid")
        print(f"  Detalhe: {_ERRO}")
        print("═" * 72 + "\n")
        raise SystemExit(1)


# ── ACOPLAMENTO: fidelidade da rede → ruído do circuito ──────────────────────
def fidelidade_para_ruido(F):
    """
    Converte fidelidade F no parâmetro p do canal despolarizante (PennyLane).
    Relação rigorosa (1 qubit de um par de Bell sob despolarização): F = 1 − p.
    Logo, p = 1 − F. (Para a aproximação antiga use (4/3)(1−F).)
    """
    return min(max(1.0 - F, 0.0), 0.75)


# ── Simulação de um enlace de fibra ──────────────────────────────────────────
def simular_link_netsquid(distancia_km, depolar_rate=200.0,
                          p_loss_init=0.0, p_loss_length=0.2, n_runs=80):
    """
    Distribui um par de Bell por um canal de fibra ruidoso e estima
    (fidelidade_media, latencia_s, qubits_perdidos) sobre n_runs repetições.
    """
    ns.set_qstate_formalism(ns.QFormalism.DM)
    fids, delays, perdidos = [], [], 0
    for _ in range(n_runs):
        ns.sim_reset()
        canal = QuantumChannel(
            name=f"fibra_{distancia_km}km", length=distancia_km,
            models={
                "delay_model":         FibreDelayModel(),
                "quantum_loss_model":  FibreLossModel(p_loss_init=p_loss_init,
                                                      p_loss_length=p_loss_length),
                "quantum_noise_model": DepolarNoiseModel(depolar_rate=depolar_rate),
            })
        q0, q1 = qapi.create_qubits(2)
        qapi.operate(q0, ns.H)
        qapi.operate([q0, q1], ns.CNOT)
        canal.send([q1]); ns.sim_run()
        recebidos, atraso = canal.receive()
        delays.append(float(atraso) * 1e-9)
        if len(recebidos) == 0:
            fids.append(0.25); perdidos += 1
        else:
            fids.append(float(qapi.fidelity([q0, recebidos[0]], ks.b00, squared=True)))
    return sum(fids) / len(fids), sum(delays) / len(delays), perdidos


# ── Levantamento de toda a topologia ─────────────────────────────────────────
def levantar_rede(cfg):
    """
    Mede todos os enlaces da topologia e retorna a lista `rede` de dicionários:
    {no, local, dist, fid, lat, p, shots}. O orçamento de shots cai com a fidelidade.
    """
    exigir_netsquid()
    rede = []
    for no, local, d in cfg.TOPOLOGIA:
        fid, lat, perd = simular_link_netsquid(d, n_runs=cfg.N_RUNS_NET)
        p = fidelidade_para_ruido(fid)
        shots = max(8, int(cfg.BASE_SHOTS * fid ** 3))
        rede.append({"no": no, "local": local, "dist": d, "fid": fid,
                     "lat": lat, "p": p, "shots": shots, "perdidos": perd})
    return rede


def rede_uniforme(fid, shots, n=4):
    """Rede 'fictícia' com todos os nós iguais — usada nas varreduras controladas."""
    p = fidelidade_para_ruido(fid)
    return [{"no": f"n{i}", "fid": fid, "p": p, "shots": shots} for i in range(n)]
