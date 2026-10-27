"""
qfl_innetwork.federated  —  APRENDIZADO FEDERADO
================================================
Três peças plugáveis + o laço federado:

  • Transmissão (como os parâmetros viajam pela rede):
        BaseTransmission → IdealTransmission, QuantumStateTransmission
  • Agregação (como o hub combina os parâmetros):
        BaseAggregator → FedAvg  (exemplos: FedMedian, FedTrimmedMean)
  • run_federated(...) : executa o laço usando a transmissão e o agregador dados.

PARA TESTAR OUTRA ESTRATÉGIA FEDERADA: crie uma subclasse de BaseAggregator
(ex.: mediana, média aparada, FedProx) e injete no main — nada mais muda.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


# ════════════════════════════════════════════════════════════════════════════
#  TRANSMISSÃO DOS PARÂMETROS PELA REDE
# ════════════════════════════════════════════════════════════════════════════
class BaseTransmission(ABC):
    @abstractmethod
    def transmit(self, weights, node):
        """Recebe os pesos de um nó e devolve os pesos como chegam ao hub."""


class IdealTransmission(BaseTransmission):
    """Transmissão perfeita (referência, sem ruído de rede)."""
    nome = "ideal"
    def transmit(self, weights, node):
        return np.asarray(weights).copy()


class QuantumStateTransmission(BaseTransmission):
    """
    Agregação via ESTADOS QUÂNTICOS:
      codifica cada parâmetro θ em RY(θ)|0⟩ → canal despolarizante p (do enlace)
      → mede ⟨Z⟩,⟨X⟩ com `shots` finitos → recupera θ = atan2(⟨X⟩,⟨Z⟩).
    Lê node['p'] e node['shots'].
    """
    nome = "quantica"
    def transmit(self, weights, node):
        p, shots = node["p"], node["shots"]
        if not shots:
            return np.asarray(weights).copy()
        flat = np.asarray(weights).ravel()
        thw = np.arctan2(np.sin(flat), np.cos(flat))
        dev = qml.device("default.mixed", wires=1, shots=shots)

        @qml.qnode(dev)
        def mz(t):
            qml.RY(t, wires=0); qml.DepolarizingChannel(p, wires=0)
            return qml.expval(qml.PauliZ(0))

        @qml.qnode(dev)
        def mx(t):
            qml.RY(t, wires=0); qml.DepolarizingChannel(p, wires=0)
            return qml.expval(qml.PauliX(0))

        z = np.array(mz(pnp.array(thw)))
        x = np.array(mx(pnp.array(thw)))
        return np.arctan2(x, z).reshape(np.asarray(weights).shape)


# ════════════════════════════════════════════════════════════════════════════
#  ESTRATÉGIAS DE AGREGAÇÃO
# ════════════════════════════════════════════════════════════════════════════
class BaseAggregator(ABC):
    nome = "base"
    @abstractmethod
    def aggregate(self, lista_pesos, tamanhos=None):
        """Combina a lista de pesos dos nós num único tensor global."""


class FedAvg(BaseAggregator):
    """Média (ponderada pelo tamanho do dado local, se fornecido)."""
    nome = "FedAvg"
    def aggregate(self, lista_pesos, tamanhos=None):
        arr = np.array(lista_pesos)
        if tamanhos is None:
            return arr.mean(axis=0)
        w = np.array(tamanhos, dtype=float); w /= w.sum()
        return np.tensordot(w, arr, axes=(0, 0))


class FedMedian(BaseAggregator):
    """Mediana elemento a elemento — exemplo de estratégia robusta a outliers."""
    nome = "FedMedian"
    def aggregate(self, lista_pesos, tamanhos=None):
        return np.median(np.array(lista_pesos), axis=0)


# ════════════════════════════════════════════════════════════════════════════
#  LAÇO FEDERADO
# ════════════════════════════════════════════════════════════════════════════
def erro_angular_abs(a, b):
    """Erros angulares absolutos elemento a elemento (lida com o 'embrulho' 2π)."""
    a = np.arctan2(np.sin(a), np.cos(a))
    b = np.arctan2(np.sin(b), np.cos(b))
    return np.abs(np.arctan2(np.sin(a - b), np.cos(a - b)))


def run_federated(model, rede, particoes, Xte, yte, w0,
                  transmission, aggregator, rounds, epocas_loc, lr,
                  coletar_erros=False):
    """
    Executa o QFL.
      model        : instância de BaseQMLModel
      rede         : lista de nós (cada um com 'p' e 'shots' se quântica)
      particoes    : lista de (X, y) por nó (dados locais)
      transmission : BaseTransmission (ideal ou quântica)
      aggregator   : BaseAggregator (FedAvg, FedMedian, ...)
    Retorna a lista de acurácias por rodada (e, opcional, erros por nó).
    """
    wg = np.array(w0); bg = 0.0
    hist = []
    erros = {r["no"]: [] for r in rede}
    for _ in range(rounds):
        ws, bs = [], []
        for r, (Xi, yi) in zip(rede, particoes):
            wl, bl = model.train_local(wg, Xi, yi, epocas_loc, lr)
            rx = transmission.transmit(wl, r)
            if coletar_erros:
                erros[r["no"]].extend(erro_angular_abs(wl.ravel(), rx.ravel()).tolist())
            ws.append(rx); bs.append(bl)
        wg = aggregator.aggregate(ws)          # estratégia federada plugável
        bg = float(np.mean(bs))
        hist.append(model.accuracy(Xte, yte, wg, bg))
    return (hist, erros) if coletar_erros else hist


def treino_centralizado(model, particoes, Xte, yte, w0, epocas, lr):
    """Baseline centralizado: treina na união de todos os dados."""
    X = np.vstack([p[0] for p in particoes])
    y = np.concatenate([p[1] for p in particoes])
    w, b = model.train_local(w0, X, y, epocas, lr)
    return model.accuracy(Xte, yte, w, b)
