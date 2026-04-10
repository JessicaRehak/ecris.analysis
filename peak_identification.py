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
plt.ion()


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
mq = csd.m_over_q
bc = csd.beam_current
if mq is not None and bc is not None:
    plt.plot(mq, bc)
plt.grid()


def find_element_peaks(peaks, csd, m):
    # Identify which peaks match potential m/q values
    peaks = np.array(peaks)
    if csd.m_over_q is None:
        return np.array([], dtype=int)

    mq_measured = csd.m_over_q[peaks]
    charge = m / mq_measured
    # Check if charge is close to an integer
    peak_mask = ~(((charge % 1) > 0.05) & ((charge % 1) < 0.95))
    return peaks[peak_mask]


from dataclasses import dataclass
from typing import Any, List, Set, cast


@dataclass
class ElementEvaluation:
    s: str
    m: int
    z: int
    a: float
    m_over_q: np.ndarray
    current: np.ndarray
    peak_indices: np.ndarray

    def symbol(self):
        return f"{self.s}-{self.m}"


def get_evals(peaks, csd, isotopes_to_exclude, identified_peak_indices, min_abundance=25):
    evaluations = []
    excluded_z = [v[0] for v in isotopes_to_exclude]
    excluded_m = [v[1] for v in isotopes_to_exclude]

    # Filter isotopes by abundance
    candidate_isotopes = isotopes[isotopes["a"] > min_abundance]

    for _, isotope in candidate_isotopes.iterrows():
        s, m, z, a = isotope.s, int(np.round(isotope.m)), isotope.z, isotope.a
        if z in excluded_z and m in excluded_m:
            continue

        found_peaks = find_element_peaks(peaks, csd, m)
        if len(found_peaks) == 0:
            continue

        # If all found peaks are already identified, skip unless it's a known persistent element
        if all(p in identified_peak_indices for p in found_peaks):
            continue

        evaluations.append(
            ElementEvaluation(
                s,
                m,
                z,
                a,
                csd.m_over_q[found_peaks] if csd.m_over_q is not None else np.array([]),
                csd.beam_current[found_peaks] if csd.beam_current is not None else np.array([]),
                found_peaks,
            )
        )
    return evaluations


from scipy.signal import find_peaks

# Find all peaks
if csd.beam_current is not None:
    peaks = find_peaks(csd.beam_current.clip(5))[0]
else:
    peaks = np.array([], dtype=int)

# 1. Finds O, N, C peaks
persistent_elements = [
    {"name": "O", "z": 8, "m": 16},
    {"name": "N", "z": 7, "m": 14},
    {"name": "C", "z": 6, "m": 12},
]

# 2. Ask the user for any elements they know are present
print("Current persistent elements: O, N, C")
known_elements_input = input(
    "Enter any additional known element symbols (e.g., 'Cl,Ar', or with mass 'O-18', or empty to skip): "
)
known_symbols = [s.strip() for s in known_elements_input.split(",") if s.strip()]

for item in known_symbols:
    if "-" in item:
        parts = item.split("-")
        symbol = parts[0]
        try:
            mass_num = int(parts[1])
            matches = isotopes[(isotopes["s"] == symbol) & (isotopes.index == mass_num)]
        except (ValueError, IndexError):
            print(f"Warning: Invalid format or mass number for '{item}'.")
            continue
    else:
        symbol = item
        matches = isotopes[isotopes["s"] == symbol]

    if matches.empty:
        print(f"Warning: Symbol '{item}' not found in isotopes data.")
        continue

    # Find most abundant among matches
    abundance_values = np.array(matches["a"])
    max_idx = int(abundance_values.argmax())
    most_abundant = matches.iloc[max_idx]
    persistent_elements.append(
        {
            "name": str(most_abundant["s"]),
            "z": int(most_abundant["z"]),
            "m": int(np.round(float(cast(Any, most_abundant["m"])))),
        }
    )

# 3. Find initial identified peaks
identified_evaluations = []
identified_peak_indices = set()
isotopes_to_exclude = []

for elem in persistent_elements:
    found_peaks = find_element_peaks(peaks, csd, elem["m"])
    if len(found_peaks) > 0:
        identified_peak_indices.update(found_peaks)
        ev = ElementEvaluation(
            str(elem["name"]),
            int(elem["m"]),
            int(elem["z"]),
            100.0,
            csd.m_over_q[found_peaks] if csd.m_over_q is not None else np.array([]),
            csd.beam_current[found_peaks] if csd.beam_current is not None else np.array([]),
            found_peaks,
        )
        identified_evaluations.append(ev)
        isotopes_to_exclude.append((elem["z"], elem["m"]))


def refresh_plot(csd, identified_evals, candidate_eval=None, highlight_label=None):
    plt.close("all")  # Ensure only one plot is open
    clear_output(wait=True)
    fig, ax = make_plot(r"$m/q$", r"$I (\mu A)$")
    if csd.m_over_q is not None and csd.beam_current is not None:
        ax.plot(csd.m_over_q, csd.beam_current, "--", color="gray", alpha=0.5)

    for ev in identified_evals:
        ax.plot(ev.m_over_q, ev.current, "x", label=ev.symbol())

    if candidate_eval:
        label = highlight_label if highlight_label else f"CANDIDATE: {candidate_eval.symbol()}"
        ax.plot(
            candidate_eval.m_over_q,
            candidate_eval.current,
            "v",
            markersize=10,
            label=label,
        )

    ax.legend()
    plt.draw()
    plt.pause(0.1)


# 4. Interactive identification loop
maybe_evaluations = []
mq_input = ""
while True:
    refresh_plot(csd, identified_evaluations)
    mq_input = (
        input("\nEnter m/q to investigate (or 'f' for final evaluation, 'q' to quit): ")
        .strip()
        .lower()
    )

    if mq_input == "q":
        break
    if mq_input == "f":
        break

    try:
        target_mq = float(mq_input)
    except ValueError:
        print("Invalid input. Please enter a number.")
        continue

    # Verification step
    suggested_candidates = []
    peak_mqs = csd.m_over_q[peaks] if csd.m_over_q is not None else np.array([])
    if len(peak_mqs) > 0:
        nearest_idx = peaks[np.argmin(np.abs(peak_mqs - target_mq))]
        peak_mq = float(csd.m_over_q[nearest_idx]) if csd.m_over_q is not None else 0.0
        peak_current = (
            float(csd.beam_current[nearest_idx]) if csd.beam_current is not None else 0.0
        )

        # Show target marker
        verify_ev = ElementEvaluation(
            "TARGET",
            0,
            0,
            0.0,
            np.array([peak_mq]),
            np.array([peak_current]),
            np.array([int(nearest_idx)]),
        )
        refresh_plot(csd, identified_evaluations, verify_ev, highlight_label="TARGET PEAK")

        ans = (
            input(
                f"Investigate peak at m/q = {peak_mq:.3f}? (y: yes, n: no/retry, q: quit, or enter symbol like 'Ar-40'): "
            )
            .strip()
            .lower()
        )
        if ans == "n":
            continue
        if ans == "q":
            break

        # Check for user suggestion
        if ans != "y":
            suggestion = ans.upper()
            if "-" in suggestion:
                parts = suggestion.split("-")
                symbol = parts[0]
                try:
                    mass_num = int(parts[1])
                    matches = isotopes[(isotopes["s"] == symbol) & (isotopes.index == mass_num)]
                except (ValueError, IndexError):
                    matches = isotopes[0:0]
            else:
                symbol = suggestion
                matches = isotopes[isotopes["s"] == symbol]

            for _, isotope in matches.iterrows():
                mass = float(cast(Any, isotope["m"]))
                best_q = int(np.round(mass / peak_mq))
                if 1 <= best_q <= int(cast(Any, isotope["z"])):
                    found_peaks = find_element_peaks(peaks, csd, mass)
                    ev = ElementEvaluation(
                        str(isotope["s"]),
                        int(np.round(mass)),
                        int(cast(Any, isotope["z"])),
                        float(cast(Any, isotope["a"])),
                        csd.m_over_q[found_peaks] if csd.m_over_q is not None else np.array([]),
                        csd.beam_current[found_peaks]
                        if csd.beam_current is not None
                        else np.array([]),
                        found_peaks,
                    )
                    suggested_candidates.append(ev)

        target_mq = peak_mq  # Use actual peak m/q for candidate matching

    # Find candidates near this m/q
    candidates = []
    for charge in range(1, 31):  # Search up to q=30
        target_mass = target_mq * charge
        # Find isotopes with mass near target_mass
        matches = isotopes[
            (isotopes["m"] > target_mass - 0.5) & (isotopes["m"] < target_mass + 0.5)
        ]
        for _, isotope in matches.iterrows():
            z_val = int(cast(Any, isotope["z"]))
            if charge > z_val:
                continue  # Skip non-physical candidates (charge > atomic number)

            found_peaks = find_element_peaks(peaks, csd, isotope["m"])
            if len(found_peaks) > 0:
                if any(
                    np.abs(csd.m_over_q[p] - target_mq) < 0.2
                    for p in found_peaks
                    if csd.m_over_q is not None
                ):
                    # Extract values explicitly as scalars
                    s_val = str(isotope["s"])
                    m_val = int(np.round(float(cast(Any, isotope["m"]))))
                    z_val = int(cast(Any, isotope["z"]))
                    a_val = float(cast(Any, isotope["a"]))

                    ev = ElementEvaluation(
                        s_val,
                        m_val,
                        z_val,
                        a_val,
                        csd.m_over_q[found_peaks] if csd.m_over_q is not None else np.array([]),
                        csd.beam_current[found_peaks]
                        if csd.beam_current is not None
                        else np.array([]),
                        found_peaks,
                    )
                    if not any(c.symbol() == ev.symbol() for c in candidates):
                        candidates.append(ev)

    if not candidates and not suggested_candidates:
        print(f"No candidates found for m/q {target_mq}.")
        input("Press Enter to continue...")
        continue

    # Prioritize suggested candidates and sort the rest
    candidates.sort(key=lambda x: (len(x.peak_indices), x.a), reverse=True)

    # Merge suggested and found candidates, removing duplicates
    final_candidates = suggested_candidates.copy()
    for cand in candidates:
        if not any(sc.symbol() == cand.symbol() for sc in suggested_candidates):
            final_candidates.append(cand)

    idx = 0
    while idx < len(final_candidates):
        cand = final_candidates[idx]
        refresh_plot(csd, identified_evaluations, cand)
        ans = (
            input(f"Accept {cand.symbol()}? (y: yes, n: no, m: maybe, s: skip/next): ")
            .strip()
            .lower()
        )

        if ans == "y":
            identified_peak_indices.update(cand.peak_indices)
            identified_evaluations.append(cand)
            break
        elif ans == "m":
            maybe_evaluations.append(cand)
            break
        elif ans == "n":
            idx += 1
        elif ans == "s":
            idx += 1
        else:
            print("Invalid input. Use y/n/m/s.")

# 5. Final evaluation review
if mq_input != "q":
    clear_output(wait=True)
    print("Final Evaluation Review")
    print("-----------------------")
    all_options = identified_evaluations + maybe_evaluations
    unique_options = {ev.symbol(): ev for ev in all_options}.values()

    final_selections = []
    options_list = list(unique_options)
    if options_list:
        print("Select elements to include in final evaluation (comma-separated indices):")
        for i, ev in enumerate(options_list):
            status = "Identified" if ev in identified_evaluations else "Maybe"
            print(f"{i}: {ev.symbol()} (Abundance: {ev.a:.1f}%, Status: {status})")

        sel_input = input("Selections (empty for defaults): ").strip()
        if sel_input:
            try:
                indices = [int(x.strip()) for x in sel_input.split(",") if x.strip().isdigit()]
                final_selections = [options_list[i] for i in indices if 0 <= i < len(options_list)]
            except (ValueError, IndexError):
                print("Invalid selections, using current identified list.")
                final_selections = identified_evaluations
        else:
            final_selections = identified_evaluations
    else:
        final_selections = identified_evaluations

    print("\nFinal Elements for Evaluation:")
    for ev in final_selections:
        print(f"- {ev.symbol()} (Mass: {ev.m}, Z: {ev.z})")

    refresh_plot(csd, final_selections)
    print("\nEvaluation complete.")
    plt.show(block=True)
