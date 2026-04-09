import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from ops.ecris.analysis.io import read_csd_from_file_pair
from ops.ecris.analysis.csd.polynomial_fit import polynomial_fit_mq
from ops.ecris.analysis.csd.m_over_q import estimate_m_over_q
from ops.ecris.analysis.model.element import Element
import pandas as pd
from IPython.display import clear_output

isotopes = pd.read_csv("./data/IsotopeData.txt", delimiter="\\s+", names=["s", "z", "a", "m"])

plt.rcParams.update(
    {
        "text.usetex": True,
    }
)


def make_plot(x_label, y_label):
    fig = plt.figure(figsize=(9, 6))
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(alpha=0.5, ls="--")
    return fig, ax


csd_path = Path("/home/work/repos/ops/ecris.analysis/data/csds/csd_1762894074")
print(f"{csd_path.exists()=}")

csd = read_csd_from_file_pair(csd_path)
csd.m_over_q, solution = polynomial_fit_mq(
    csd,
    [Element("O", "Oxygen", 16, 8)],
    polynomial_order=4,
    always_optimize=True,
    nonlinear_bounds=(-1e-2, 1e-2),
)
plt.plot(csd.m_over_q, csd.beam_current)
plt.grid()


def find_element_peaks(peaks, csd, m):
    # Identify which peaks match potential m/q values
    peaks = np.array(peaks)
    q = 1 / csd.m_over_q[peaks] * m
    peak_indices = ~(((q % 1) > 0.01) & ((q % 1) < 0.99))
    mask_array = [False] * len(csd.m_over_q)
    for i, peak in enumerate(peaks):
        if peak_indices[i]:
            mask_array[peak] = True
    return mask_array


from dataclasses import dataclass
from typing import List


@dataclass
class ElementEvaluation:
    s: str
    m: int
    z: int
    a: float
    m_over_q: np.ndarray
    current: np.ndarray
    indices: List[int]

    def symbol(self):
        return f"{self.s}-{self.m}"


def get_evals(peaks, csd, isotopes_to_exclude, identified_peaks, min_abundance=25):
    evaluations = []
    excluded_z = [v[0] for v in isotopes_to_exclude]
    excluded_m = [v[1] for v in isotopes_to_exclude]

    for isotope in isotopes[isotopes["a"] > min_abundance].iterrows():
        isotope = isotope[1]
        s, m, z, a = isotope.s, int(np.round(isotope.m)), isotope.z, isotope.a
        if z in excluded_z and m in excluded_m:
            continue
        peak_indices = find_element_peaks(peaks, csd, m)
        # if sorted(set(cut_indexes + identified)) == sorted(set(identified)):
        # continue
        evaluations.append(
            ElementEvaluation(
                s,
                m,
                z,
                a,
                csd.m_over_q[peak_indices],
                csd.beam_current[peak_indices],
                peak_indices,
            )
        )
    return evaluations


from scipy.signal import find_peaks

# Find all peaks
peaks = find_peaks(csd.beam_current.clip(5))[0]

# Find known peaks
peak_evaluations = []
identified = set()
for m, z, name in zip([16, 14, 12, 35], [8, 7, 6, 17], ["O", "N", "C", "Cl"]):
    peak_indices = find_element_peaks(peaks, csd, m)
    identified.update(peak_indices)
    peak_evaluations.append(
        ElementEvaluation(
            name,
            m,
            z,
            100,
            csd.m_over_q[peak_indices],
            csd.beam_current[peak_indices],
            peak_indices,
        )
    )

isotopes_to_exclude = [(8, 16), (6, 12), (7, 14), (17, 35)]

run = True
while run:
    i = 0
    evaluations = get_evals(peaks, csd, isotopes_to_exclude, identified)
    sorted_evals = sorted(evaluations, key=lambda e: len(e.m_over_q), reverse=True)
    while True:
        evals_to_consider = sorted_evals[i:]
        sorted_by_abundance = sorted(
            [
                e
                for e in evals_to_consider
                if len(e.m_over_q) == len(evals_to_consider[0].m_over_q)
            ],
            key=lambda e: e.a,
        )
        next_eval = sorted_by_abundance[0]
        clear_output(wait=True)
        make_plot(r"$m/q$", r"$I (\mu A)$")
        plt.plot(csd.m_over_q, csd.beam_current, "--")
        for peak in peak_evaluations:
            plt.plot(peak.m_over_q, peak.current, "x", label=peak.symbol())
        plt.plot(next_eval.m_over_q, next_eval.current, "v", label=next_eval.symbol())
        plt.legend()
        plt.show()
        accept = input("Accept isotope?")
        match accept:
            case "q":
                run = False
                break
            case "y":
                identified.update(next_eval.indices)
                peak_evaluations.append(next_eval)
                isotopes_to_exclude.append((next_eval.z, next_eval.m))
                i += 1
                break
            case _:
                i += 1
                continue
