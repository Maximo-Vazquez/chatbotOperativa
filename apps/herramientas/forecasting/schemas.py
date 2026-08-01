"""Estructuras de datos internas del nucleo de pronostico.

``engine.py`` devuelve estas dataclasses como resultado normalizado de un
ajuste. Las herramientas publicas (``tools/modelo_arima.py`` y, en fases
futuras, MA/SARIMA/ARIMAX/SARIMAX) las convierten al diccionario
JSON-serializable que consume el chatbot. Mantener esta capa separada evita
que la logica estadistica y el contrato JSON externo queden acoplados.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ParametroEstimado:
    """Un coeficiente estimado por el modelo, con su informacion de inferencia."""

    nombre: str
    tipo: str
    coeficiente: float
    error_estandar: Optional[float]
    estadistico: Optional[float]
    p_value: Optional[float]
    intervalo_confianza: Optional[tuple[float, float]]
    significativo_005: Optional[bool]


@dataclass
class PronosticoPaso:
    """El pronostico puntual y su intervalo para un paso futuro."""

    paso: int
    pronostico: float
    limite_inferior: float
    limite_superior: float


@dataclass
class ResultadoAjusteARIMA:
    """Resultado normalizado de ajustar un modelo ARIMA(p, d, q), opcionalmente
    con componente estacional (SARIMA) via `seasonal_order=(P,D,Q,s)`."""

    modelo: str
    orden: tuple[int, int, int]
    orden_estacional: Optional[tuple[int, int, int, int]]
    trend: str
    descripcion_tendencia: str
    n_observaciones: int
    n_observaciones_efectivas: Optional[int]
    parametros: list[ParametroEstimado]
    aic: Optional[float]
    bic: Optional[float]
    log_likelihood: Optional[float]
    convergio: Optional[bool]
    enforce_stationarity: bool
    enforce_invertibility: bool
    residuos: np.ndarray
    pronosticos: list[PronosticoPaso]
    nivel_confianza: float
    advertencias: list[dict]
