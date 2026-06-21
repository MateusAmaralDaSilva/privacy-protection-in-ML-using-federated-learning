import numpy as np


class CrostonForecaster:
    """
    Método de Croston para previsão de demanda intermitente.

    Separa a série em dois componentes, cada um suavizado de forma independente:
        l_t  — nível da demanda (tamanho das demandas não-nulas)
        p_t  — intervalo entre demandas (períodos entre ocorrências não-nulas)

    Previsão:        l_t / p_t
    Variante SBA:    (1 - beta/2) * l_t / p_t   [Syntetos-Boylan, reduz viés]

    Estado federado: np.ndarray([l, p], dtype=float64) — dois escalares por cliente.
    O servidor agrega via FedAvg ponderado pelo número de amostras.
    """

    def __init__(self, alpha: float = 0.1, beta: float = 0.1, variant: str = "sba"):
        if not (0 < alpha <= 1):
            raise ValueError("alpha deve estar em (0, 1]")
        if not (0 < beta <= 1):
            raise ValueError("beta deve estar em (0, 1]")
        self.alpha = alpha
        self.beta = beta
        self.variant = variant
        self.demand_level: float = 0.0
        self.interval: float = 1.0
        self.fitted: bool = False

    # ------------------------------------------------------------------
    # Algoritmo principal
    # ------------------------------------------------------------------

    def fit(
        self,
        y: np.ndarray,
        init_demand_level: float | None = None,
        init_interval: float | None = None,
    ) -> "CrostonForecaster":
        """
        Ajusta o modelo a uma série de demanda diária não-negativa.

        Parameters
        ----------
        y : np.ndarray
            Série de demanda bruta (pode conter zeros).
        init_demand_level : float, opcional
            Warm-start: nível de demanda recebido do modelo global federado.
            Quando fornecido junto com init_interval, a suavização parte desses
            valores em vez de ser inicializada pelo primeiro valor não-nulo local.
        init_interval : float, opcional
            Warm-start: intervalo entre demandas recebido do modelo global.
        """
        y = np.asarray(y, dtype=np.float64)
        non_zero_idx = np.flatnonzero(y > 0)

        has_warm_start = (
            init_demand_level is not None and init_interval is not None
        )

        if len(non_zero_idx) == 0:
            # Partição sem demanda positiva — mantém o estado global ou usa fallback
            self.demand_level = float(init_demand_level) if has_warm_start else 0.0
            self.interval = float(init_interval) if has_warm_start else float(len(y))
            self.fitted = True
            return self

        if has_warm_start:
            # Continua a suavização a partir do estado global
            l = float(init_demand_level)
            p = float(init_interval)
            prev_pos = -1       # posição virtual antes do início da série
            idx_iter = non_zero_idx
        else:
            # Inicialização clássica pelo primeiro valor não-nulo
            first = int(non_zero_idx[0])
            l = float(y[first])
            p = float(first + 1)   # intervalo do período 0 até a 1ª demanda
            prev_pos = first
            idx_iter = non_zero_idx[1:]

        for raw_idx in idx_iter:
            idx = int(raw_idx)
            q = float(idx - prev_pos) if prev_pos >= 0 else float(idx + 1)
            l = self.alpha * y[idx] + (1.0 - self.alpha) * l
            p = self.beta * q + (1.0 - self.beta) * p
            prev_pos = idx

        self.demand_level = l
        self.interval = max(p, 1e-9)   # evita divisão por zero
        self.fitted = True
        return self

    # ------------------------------------------------------------------
    # Previsão
    # ------------------------------------------------------------------

    def predict(self) -> float:
        """Retorna a previsão pontual para o próximo período."""
        if not self.fitted:
            raise RuntimeError("Chame fit() antes de predict().")
        base = self.demand_level / self.interval
        if self.variant == "sba":
            return (1.0 - self.beta / 2.0) * base
        return base

    # ------------------------------------------------------------------
    # Avaliação
    # ------------------------------------------------------------------

    def evaluate(self, y_test: np.ndarray) -> tuple[float, float, float, float]:
        """
        Avalia o modelo em uma série de teste.
        A previsão de Croston é constante ao longo do horizonte.

        Returns
        -------
        (mse, mae, rmse, r2) : tuple[float, float, float, float]
        """
        forecast = self.predict()
        y_test = np.asarray(y_test, dtype=np.float64)
        errors = y_test - forecast
        ss_res = float(np.sum(errors ** 2))
        ss_tot = float(np.sum((y_test - np.mean(y_test)) ** 2))
        mse  = ss_res / len(y_test)
        mae  = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(mse))
        r2   = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return mse, mae, rmse, r2

    # ------------------------------------------------------------------
    # Serialização para o Flower
    # ------------------------------------------------------------------

    def get_state(self) -> np.ndarray:
        """Retorna [demand_level, interval] como ndarray float64 (2 elementos)."""
        return np.array([self.demand_level, self.interval], dtype=np.float64)

    def set_state(self, state: np.ndarray) -> None:
        """Restaura o estado a partir de um ndarray serializado."""
        self.demand_level = float(state[0])
        self.interval = float(state[1])
        self.fitted = True
