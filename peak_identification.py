import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, List, Set, cast, Dict, Optional, Tuple
from dataclasses import dataclass
from scipy.signal import find_peaks
from IPython.display import clear_output
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt

# Assuming these are available in the environment/package
from ops.ecris.analysis.csd.polynomial_fit import polynomial_fit_mq
from ops.ecris.analysis.model.element import Element

console = Console()


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

    def score(self, max_mq: float) -> float:
        # Calculate how many peaks we expect to see for this isotope
        # We expect one peak for each charge q from 1 to z, provided m/q is within the measured range
        expected_qs = [self.m / q for q in range(1, self.z + 1)]
        expected_count = len([mq for mq in expected_qs if mq <= max_mq])

        if expected_count == 0:
            return 0.0

        # Score is the fraction of expected peaks that were actually found
        return len(self.peak_indices) / expected_count


def find_element_peaks(peaks: np.ndarray, csd: Any, m: float) -> np.ndarray:
    """Identify which peaks match potential m/q values for a given mass."""
    if csd.m_over_q is None:
        return np.array([], dtype=int)

    mq_measured = csd.m_over_q[peaks]
    charge = m / mq_measured
    # Check if charge is close to an integer
    peak_mask = ~(((charge % 1) > 0.05) & ((charge % 1) < 0.95))
    return peaks[peak_mask]


def lookup_isotopes(query: str, isotopes_df: pd.DataFrame) -> Any:
    """Lookup isotopes for a given query like 'Ar' or 'Ar-40'."""
    query = query.strip()
    if "-" in query:
        parts = query.split("-")
        symbol = parts[0].capitalize()
        try:
            mass_num = int(parts[1])
            return isotopes_df[(isotopes_df["s"] == symbol) & (isotopes_df.index == mass_num)]
        except (ValueError, IndexError):
            return isotopes_df[0:0]
    else:
        symbol = query.capitalize()
        return isotopes_df[isotopes_df["s"] == symbol]


def create_evaluation(isotope: Any, csd: Any, peaks: np.ndarray) -> ElementEvaluation:
    """Create an ElementEvaluation for a given isotope."""
    mass = float(cast(Any, isotope["m"]))
    found_peaks = find_element_peaks(peaks, csd, mass)
    return ElementEvaluation(
        str(isotope["s"]),
        int(np.round(mass)),
        int(cast(Any, isotope["z"])),
        float(cast(Any, isotope["a"])),
        csd.m_over_q[found_peaks] if csd.m_over_q is not None else np.array([]),
        csd.beam_current[found_peaks] if csd.beam_current is not None else np.array([]),
        found_peaks,
    )


def make_plot(x_label: str, y_label: str):
    """Create a standard CSD plot axis."""
    fig = plt.figure(figsize=(9, 6))
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(alpha=0.5, ls="--")
    return fig, ax


class PeakIdentifier:
    def __init__(self, csd: Any, isotopes_df: Optional[pd.DataFrame] = None):
        self.csd = csd
        if isotopes_df is None:
            self.isotopes = pd.read_csv(
                "./data/IsotopeData.txt", delimiter="\\s+", names=["s", "z", "a", "m"]
            )
        else:
            self.isotopes = isotopes_df

        self.peaks = self._find_peaks()
        self.identified_evaluations: List[ElementEvaluation] = []
        self.maybe_evaluations: List[ElementEvaluation] = []
        self.no_evaluations: List[ElementEvaluation] = []
        self.identified_peak_indices: Set[int] = set()

    def _find_peaks(self) -> np.ndarray:
        if self.csd.beam_current is not None:
            return find_peaks(self.csd.beam_current.clip(2))[0]
        return np.array([], dtype=int)

    def refresh_plot(
        self,
        candidate_eval: Optional[ElementEvaluation] = None,
        highlight_label: Optional[str] = None,
    ):
        plt.close("all")
        clear_output(wait=True)
        fig, ax = make_plot(r"$m/q$", r"$I (\mu A)$")

        if self.csd.m_over_q is not None and self.csd.beam_current is not None:
            ax.plot(self.csd.m_over_q, self.csd.beam_current, "--", color="gray", alpha=0.5)

        for ev in self.identified_evaluations:
            ax.plot(ev.m_over_q, ev.current, "x", label=ev.symbol())

        if candidate_eval:
            label = highlight_label if highlight_label else f"CANDIDATE: {candidate_eval.symbol()}"
            ax.plot(
                candidate_eval.m_over_q, candidate_eval.current, "v", markersize=10, label=label
            )

        ax.legend()
        plt.draw()
        plt.pause(0.1)

    def setup_persistent_elements(self, default_symbols: List[str] = ["O", "N", "C"]):
        console.print()
        console.print(Panel("[bold cyan]INITIAL SETUP: IDENTIFIED ELEMENTS[/]", expand=False))
        console.print(f"Default persistent elements: [bold green]{', '.join(default_symbols)}[/]")
        console.print()

        known_elements_input = Prompt.ask(
            "Enter any [bold cyan]additional known element symbols[/]\n"
            "  (e.g., 'Cl,Ar', or with mass 'O-18')\n"
            "  [dim](leave empty to skip)[/]",
            default="",
        )
        console.print()

        symbols_to_process = default_symbols + [
            s.strip() for s in known_elements_input.split(",") if s.strip()
        ]

        persistent_elements = []
        for item in symbols_to_process:
            matches = lookup_isotopes(item, self.isotopes)
            if matches.empty:
                console.print(
                    f"[bold red]Warning:[/] Symbol/Isotope '[bold white]{item}[/]' not found."
                )
                continue

            most_abundant = matches.iloc[matches["a"].to_numpy().argmax()]
            persistent_elements.append(
                {
                    "name": str(most_abundant["s"]),
                    "z": int(most_abundant["z"]),
                    "m": int(np.round(float(cast(Any, most_abundant["m"])))),
                }
            )

        for elem in persistent_elements:
            found_peaks = find_element_peaks(self.peaks, self.csd, elem["m"])
            if len(found_peaks) > 0:
                self.identified_peak_indices.update(found_peaks)
                ev = ElementEvaluation(
                    str(elem["name"]),
                    int(elem["m"]),
                    int(elem["z"]),
                    100.0,
                    self.csd.m_over_q[found_peaks]
                    if self.csd.m_over_q is not None
                    else np.array([]),
                    self.csd.beam_current[found_peaks]
                    if self.csd.beam_current is not None
                    else np.array([]),
                    found_peaks,
                )
                self.identified_evaluations.append(ev)

    def run_investigation(self):
        while True:
            self.refresh_plot()
            console.print()
            console.print(Panel("[bold yellow]PEAK INVESTIGATION LOOP[/]", expand=False))
            console.print("[dim](Enter an m/q value to search for matching isotopes)[/]")
            console.print()

            mq_input = Prompt.ask(
                "Enter [bold yellow]m/q[/] value\n  [cyan]f[/] : Final evaluation\n  [red]q[/] : Quit\n",
                default="f",
            ).lower()

            if mq_input == "q":
                return "quit"
            if mq_input == "f":
                break

            try:
                target_mq = float(mq_input)
            except ValueError:
                console.print("[red]Invalid input. Please enter a number.[/]")
                continue

            self._investigate_peak(target_mq)
        return "success"

    def _investigate_peak(self, target_mq: float):
        peak_mqs = self.csd.m_over_q[self.peaks] if self.csd.m_over_q is not None else np.array([])
        if len(peak_mqs) == 0:
            return

        nearest_idx = self.peaks[np.argmin(np.abs(peak_mqs - target_mq))]
        peak_mq = float(self.csd.m_over_q[nearest_idx])
        peak_current = float(self.csd.beam_current[nearest_idx])

        # Verification step
        verify_ev = ElementEvaluation(
            "TARGET",
            0,
            0,
            0.0,
            np.array([peak_mq]),
            np.array([peak_current]),
            np.array([int(nearest_idx)]),
        )
        self.refresh_plot(verify_ev, highlight_label="TARGET PEAK")

        console.print()
        console.print(f"Targeting: [bold magenta]m/q = {peak_mq:.3f}[/]")
        ans = Prompt.ask(
            "Is this the peak you want to investigate?\n"
            "  [bold]y[/] : Yes, search for isotopes\n"
            "  [bold]n[/] : No, retry m/q entry\n"
            "  [bold]q[/] : Quit\n"
            "  [bold]symbol[/] : Enter a symbol manually (e.g., 'Ar-40')\n",
            default="y",
        ).lower()
        console.print()

        if ans == "n":
            return
        if ans == "q":
            return

        suggested_candidates = []
        if ans != "y":
            matches = lookup_isotopes(ans, self.isotopes)
            for _, isotope in matches.iterrows():
                mass = float(cast(Any, isotope["m"]))
                best_q = int(np.round(mass / peak_mq))
                if 1 <= best_q <= int(cast(Any, isotope["z"])):
                    ev = create_evaluation(isotope, self.csd, self.peaks)
                    if nearest_idx in ev.peak_indices:
                        suggested_candidates.append(ev)

        # Find other candidates
        candidates = []
        for charge in range(1, 31):
            target_mass = peak_mq * charge
            matches = self.isotopes[
                (self.isotopes["m"] > target_mass - 0.5) & (self.isotopes["m"] < target_mass + 0.5)
            ]
            for _, isotope in matches.iterrows():
                if charge > int(cast(Any, isotope["z"])):
                    continue
                found_peaks = find_element_peaks(
                    self.peaks, self.csd, float(cast(Any, isotope["m"]))
                )
                if nearest_idx in found_peaks:
                    ev = create_evaluation(isotope, self.csd, self.peaks)
                    if not any(c.symbol() == ev.symbol() for c in candidates):
                        candidates.append(ev)

        if not candidates and not suggested_candidates:
            console.print(f"[bold yellow]No candidates found for m/q {peak_mq:.3f}.[/]")
            Prompt.ask("Press Enter to continue")
            return

        max_mq = float(self.csd.m_over_q.max())
        candidates.sort(key=lambda x: (x.score(max_mq), x.a), reverse=True)
        final_candidates = suggested_candidates + [
            c
            for c in candidates
            if not any(sc.symbol() == c.symbol() for sc in suggested_candidates)
        ]

        idx = 0
        while idx < len(final_candidates):
            cand = final_candidates[idx]
            self.refresh_plot(cand)
            console.print(
                Panel(
                    f"[bold cyan]INVESTIGATING m/q {peak_mq:.3f}[/]\nCandidate [bold]{idx + 1}/{len(final_candidates)}[/]: [bold green]{cand.symbol()}[/]",
                    expand=False,
                )
            )
            console.print(
                f"  [bold]y[/] : Accept {cand.symbol()}\n  [bold]n[/] : Reject\n  [bold]m[/] : Maybe\n  [bold]s[/] : Skip\n  [bold]e[/] : End\n"
            )
            ans = Prompt.ask("Action", choices=["y", "n", "m", "e", "s"], default="y").lower()
            console.print()

            if ans == "y":
                self.identified_peak_indices.update(cand.peak_indices)
                self.identified_evaluations.append(cand)
                break
            elif ans == "e":
                break
            elif ans == "m":
                self.maybe_evaluations.append(cand)
                idx += 1
            elif ans == "n":
                self.no_evaluations.append(cand)
                idx += 1
            elif ans == "s":
                idx += 1

    def run_final_review(self) -> List[ElementEvaluation]:
        clear_output(wait=True)
        console.print(Panel("[bold green]Final Evaluation Review[/]"))
        all_options = self.identified_evaluations + self.maybe_evaluations
        unique_options = list({ev.symbol(): ev for ev in all_options}.values())

        if not unique_options:
            console.print("[yellow]No elements identified.[/]")
            return []

        table = Table(title="Candidates for Final Selection")
        table.add_column("Index", style="cyan", justify="right")
        table.add_column("Symbol", style="bold green")
        table.add_column("Score", style="magenta", justify="right")
        table.add_column("Abundance (%)", style="yellow", justify="right")
        table.add_column("Status", style="blue")

        max_mq = float(self.csd.m_over_q.max())
        for i, ev in enumerate(unique_options):
            status = "Identified" if ev in self.identified_evaluations else "Maybe"
            table.add_row(str(i), ev.symbol(), f"{ev.score(max_mq):.2f}", f"{ev.a:.1f}", status)

        console.print(table)
        sel_input = Prompt.ask(
            "Select indices for final evaluation (comma-separated or empty for defaults)",
            default="",
        )

        if sel_input:
            try:
                indices = [int(x.strip()) for x in sel_input.split(",") if x.strip().isdigit()]
                final_selections = [
                    unique_options[i] for i in indices if 0 <= i < len(unique_options)
                ]
            except:
                final_selections = self.identified_evaluations
        else:
            final_selections = self.identified_evaluations

        console.print(Panel("[bold green]Final Elements for Evaluation[/]"))
        for ev in final_selections:
            console.print(f"  • [bold green]{ev.symbol()}[/] (Mass: {ev.m}, Z: {ev.z})")

        self.refresh_plot(None)
        for ev in final_selections:
            plt.plot(ev.m_over_q, ev.current, "x", label=ev.symbol())
        plt.legend()
        plt.show(block=True)
        return final_selections


def identify_peaks(
    csd: Any, isotopes_df: Optional[pd.DataFrame] = None
) -> List[ElementEvaluation]:
    """
    Perform interactive peak identification on a CSD.

    Returns a list of selected ElementEvaluation objects.
    """
    session = PeakIdentifier(csd, isotopes_df)
    session.setup_persistent_elements()
    status = session.run_investigation()
    if status == "quit":
        return []
    return session.run_final_review()


if __name__ == "__main__":
    # Example usage
    from ops.ecris.analysis.io import read_csd_from_file_pair

    csd_path = Path("/home/work/repos/ops/ecris.analysis/data/csds/csd_1762894074")
    if csd_path.exists():
        csd = read_csd_from_file_pair(csd_path)
        # Initial fit for m/q estimation if needed
        csd.m_over_q, _ = polynomial_fit_mq(
            csd, [Element("O", "Oxygen", 16, 8)], polynomial_order=4
        )
        results = identify_peaks(csd)
        print(f"Identified {len(results)} elements.")
