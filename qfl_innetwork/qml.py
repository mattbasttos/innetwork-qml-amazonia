"""
qfl_innetwork.qml  —  PENNYLANE (modelos de aprendizado quântico)
================================================================
Interface comum (`BaseQMLModel`) + implementação `VQCModel`.

PARA TESTAR OUTRO MODELO (ex.: uma QNN com outro ansatz / encoding):
  1. Crie uma subclasse de BaseQMLModel.
  2. Implemente `init_weights`, `_circuito`, `train_local`, `predict`, `accuracy`
     (ou reaproveite os genéricos, sobrescrevendo apenas `_circuito` e `n_weights`).
  3. Aponte `MODELO` no Config e injete sua classe no main.
Veja o esqueleto `QNNModel` no fim do arquivo.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


# ════════════════════════════════════════════════════════════════════════════
#  INTERFACE COMUM
# ════════════════════════════════════════════════════════════════════════════
class BaseQMLModel(ABC):
    """Contrato que todo modelo de QML federável deve cumprir."""

    def __init__(self, n_qubits, n_layers):
        self.n_qubits = n_qubits
        self.n_layers = n_layers

    @property
    @abstractmethod
    def shape_pesos(self):
        """Formato do tensor de pesos transmitidos pela rede (como ângulos)."""

    @abstractmethod
    def init_weights(self, seed):
        """Pesos iniciais (np.ndarray)."""

    @abstractmethod
    def forward(self, X, weights):
        """Saída crua do circuito (⟨Z₀⟩) para um lote X."""

    @abstractmethod
    def train_local(self, weights0, X, y, epocas, lr):
        """Treina localmente a partir de weights0. Retorna (weights, bias)."""

    def predict(self, X, weights, bias):
        out = np.array(self.forward(np.asarray(X), weights)) + bias
        return np.sign(out)

    def accuracy(self, X, y, weights, bias):
        return float(np.mean(self.predict(X, weights, bias) == np.asarray(y)))


# ════════════════════════════════════════════════════════════════════════════
#  MODELO 1 — VQC com StronglyEntanglingLayers (padrão)
# ════════════════════════════════════════════════════════════════════════════
class VQCModel(BaseQMLModel):
    """
    Classificador Quântico Variacional:
      AngleEmbedding(RY) → StronglyEntanglingLayers → ⟨Z₀⟩.
    Treino em default.qubit (estado puro, backprop).
    """

    def __init__(self, n_qubits=4, n_layers=4):
        super().__init__(n_qubits, n_layers)
        self._dev = qml.device("default.qubit", wires=n_qubits)
        self._qnode = qml.QNode(self._circuito, self._dev, diff_method="backprop")

    @property
    def shape_pesos(self):
        return qml.StronglyEntanglingLayers.shape(self.n_layers, self.n_qubits)

    def _circuito(self, x, w):
        qml.AngleEmbedding(x, wires=range(self.n_qubits), rotation="Y")
        qml.StronglyEntanglingLayers(w, wires=range(self.n_qubits))
        return qml.expval(qml.PauliZ(0))

    def init_weights(self, seed):
        rng = np.random.default_rng(seed)
        return 0.1 * rng.standard_normal(self.shape_pesos)

    def forward(self, X, weights):
        return self._qnode(pnp.array(X, requires_grad=False), weights)

    def train_local(self, weights0, X, y, epocas, lr):
        w = pnp.array(weights0, requires_grad=True)
        b = pnp.array(0.0, requires_grad=True)
        opt = qml.AdamOptimizer(lr)
        Xt = pnp.array(X, requires_grad=False)
        yt = pnp.array(y, requires_grad=False)
        for _ in range(epocas):
            (w, b), _ = opt.step_and_cost(
                lambda w, b: pnp.mean((self._qnode(Xt, w) + b - yt) ** 2), w, b)
        return np.array(w), float(b)


# ════════════════════════════════════════════════════════════════════════════
#  MODELO 2 — ESQUELETO PARA UMA QNN (exemplo de extensão)
# ════════════════════════════════════════════════════════════════════════════
class QNNModel(VQCModel):
    """
    Exemplo de como plugar outro ansatz/encoding. Aqui herda do VQC e só troca
    o circuito por uma codificação com RE-UPLOADING de dados (data re-uploading),
    típica de QNNs. Use como ponto de partida para os seus experimentos.
    """

    @property
    def shape_pesos(self):
        # 2 blocos de re-uploading: (blocos, qubits, 3)
        return qml.StronglyEntanglingLayers.shape(self.n_layers, self.n_qubits)

    def _circuito(self, x, w):
        # data re-uploading: re-injeta os dados entre as camadas
        for L in range(self.n_layers):
            qml.AngleEmbedding(x, wires=range(self.n_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(w[L:L+1], wires=range(self.n_qubits))
        return qml.expval(qml.PauliZ(0))
