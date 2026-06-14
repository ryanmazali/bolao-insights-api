"""Calibração de probabilidades por faixa de |elo_diff| (home_elo - away_elo).

A calibração isotonic global (CalibratedClassifierCV) achata probabilidades
extremas: confrontos muito desequilibrados (ex.: Alemanha x Curaçao, elo_diff
~440) acabam com empate inflado (~23%) porque o calibrador é ajustado sobre
TODOS os confrontos, dominados por jogos equilibrados.

BinnedCalibrator treina um calibrador isotonic (one-vs-rest, por classe)
separado para cada faixa de |elo_diff|:
  - Faixa 0: elo_diff < 100   (jogos equilibrados)
  - Faixa 1: 100 <= elo_diff < 300 (favorito claro)
  - Faixa 2: elo_diff >= 300  (disparidade extrema)

Em produção, dado o |elo_diff| do confronto, aplica-se o calibrador da faixa
correspondente sobre as probabilidades brutas do modelo.
"""

import numpy as np
from sklearn.isotonic import IsotonicRegression

ELO_DIFF_BINS = (100, 300)
N_BINS = 3


def get_elo_bin(elo_diff_abs: float) -> int:
    """Retorna o índice da faixa (0, 1 ou 2) para um |elo_diff|."""
    if elo_diff_abs < ELO_DIFF_BINS[0]:
        return 0
    if elo_diff_abs < ELO_DIFF_BINS[1]:
        return 1
    return 2


class BinnedCalibrator:
    """Calibração isotonic one-vs-rest, separada por faixa de |elo_diff|."""

    def __init__(self, n_classes: int):
        self.n_classes = n_classes
        self.calibrators = {}  # bin_idx -> [IsotonicRegression por classe]

    def fit(self, raw_probs: np.ndarray, y: np.ndarray, elo_diff_abs: np.ndarray,
            min_samples: int = 50) -> "BinnedCalibrator":
        raw_probs = np.asarray(raw_probs)
        y = np.asarray(y)
        elo_diff_abs = np.asarray(elo_diff_abs)
        bins = np.array([get_elo_bin(d) for d in elo_diff_abs])

        for b in range(N_BINS):
            mask = bins == b
            # Fallback: se a faixa tiver poucas amostras, calibra com todo o
            # conjunto para evitar isotonic ajustado em poucos pontos.
            use_mask = mask if mask.sum() >= min_samples else np.ones_like(mask, dtype=bool)

            cal_list = []
            for c in range(self.n_classes):
                iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
                iso.fit(raw_probs[use_mask, c], (y[use_mask] == c).astype(float))
                cal_list.append(iso)
            self.calibrators[b] = cal_list

        return self

    def transform(self, raw_probs: np.ndarray, elo_diff_abs: np.ndarray) -> np.ndarray:
        raw_probs = np.asarray(raw_probs)
        elo_diff_abs = np.asarray(elo_diff_abs)

        out = np.zeros_like(raw_probs)
        for i in range(len(raw_probs)):
            cal_list = self.calibrators[get_elo_bin(elo_diff_abs[i])]
            for c in range(self.n_classes):
                out[i, c] = cal_list[c].predict([raw_probs[i, c]])[0]

        sums = out.sum(axis=1, keepdims=True)
        sums[sums == 0] = 1.0
        return out / sums
