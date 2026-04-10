- [x] Create a configurable list of 'persistent' elements (starting with O, N, C).
- [x] Refactor `peak_identification.py` to use this list instead of hardcoded values.
- [x] Implement a function to prompt the user for additional known elements at the start.
- [x] Fix type hinting and LSP errors (to the extent possible with dynamic types).

## 2. Interactive Plotting and User Input
- [x] Implement a mechanism to display the plot and request input simultaneously.
- [x] Use `plt.ion()` and `plt.pause()` to ensure the plot is visible during `input()`.

## 3. Targeted Identification Flow (New Loop)
- [x] Ask the user for a specific m/q value to investigate.
- [x] Filter available isotopes from `IsotopeData.txt` that could potentially match that m/q.
- [x] For each candidate:
    - [x] Display the candidate on the plot alongside current identifications.
    - [x] Prompt: "Accept [Symbol]-[Mass]? (y: yes, n: no, m: maybe, next: show another)"
    - [x] Store "yes" in `identified` and "maybe" in a `candidates` list.

## 4. Final Selection and Evaluation
- [x] Present all "yes" and "maybe" elements to the user.
- [x] Allow the user to finalize the list for the final evaluation.
- [x] Generate the final summary of identified peaks.
