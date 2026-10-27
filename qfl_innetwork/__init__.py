"""
qfl_innetwork — pacote modular de QML federado in-network (NetSquid + PennyLane).

Módulos:
  config              — SETUP (dataclass Config)
  netsquid_backend    — NETSQUID (rede quântica + acoplamento p=1−F)
  qml                 — PENNYLANE (BaseQMLModel, VQCModel, QNNModel)
  federated           — APRENDIZADO FEDERADO (transmissão + agregação + laço)
  data                — geração de dados
  experiments         — coleta de telemetria
  analysis            — teste estatístico
  export              — exportação CSV
  plots               — gráficos
"""
from .config import Config
from .qml import BaseQMLModel, VQCModel, QNNModel
from .federated import (BaseTransmission, IdealTransmission, QuantumStateTransmission,
                        BaseAggregator, FedAvg, FedMedian)

__all__ = ["Config", "BaseQMLModel", "VQCModel", "QNNModel",
           "BaseTransmission", "IdealTransmission", "QuantumStateTransmission",
           "BaseAggregator", "FedAvg", "FedMedian"]
