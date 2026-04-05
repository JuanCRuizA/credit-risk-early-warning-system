"""
Add bibliographic reference comments to code cells in NB03.
Safe-write pattern: write to .tmp, validate JSON, then rename.
"""

import json
import os
import shutil

NOTEBOOK_PATH = os.path.join(os.path.dirname(__file__), '..', 'notebooks', '03_model_training_evaluation.ipynb')
NOTEBOOK_PATH = os.path.normpath(NOTEBOOK_PATH)

# Define references: cell_index -> list of comment lines to prepend
REFERENCES = {
    16: [
        "# Reference: Fluss, Faraggi & Reiser (2005). \"Estimation of the Youden Index\n",
        "# and its Associated Cutoff Point\". Biometrical Journal, 47(4), 458-472.\n",
        "# Youden's J statistic (J = TPR - FPR) identifies the optimal threshold\n",
        "# maximizing the classifier's discriminative ability.\n",
        "\n",
    ],
    23: [
        "# Reference: Verbraken, Bravo, Weber & Baesens (2014). \"Development and application\n",
        "# of consumer credit scoring models using profit-based classification measures\".\n",
        "# European Journal of Operational Research, 238(2), 505-513.\n",
        "# Adapted profit-based framework: fixed LGD and profit margin replace the full\n",
        "# EMP integration over the loan fraction distribution.\n",
        "\n",
    ],
    30: [
        "# Reference: Beque, Coussement, Gayler & Lessmann (2017). \"Approaches for credit\n",
        "# scorecard calibration: An empirical analysis\". Knowledge-Based Systems, 134, 213-227.\n",
        "# Reliability diagram assesses calibration: predicted probabilities vs. observed\n",
        "# default rates across bins (Section 4.4 of the paper).\n",
        "\n",
    ],
    32: [
        "# Reference: Beque, Coussement, Gayler & Lessmann (2017). \"Approaches for credit\n",
        "# scorecard calibration: An empirical analysis\". Knowledge-Based Systems, 134, 213-227.\n",
        "# Brier Score (Eq. 7): BS = (1/N) * sum((y_i - p_i)^2). Measures calibration\n",
        "# quality; decomposable into uncertainty, resolution, and reliability components.\n",
        "\n",
    ],
    34: [
        "# Reference: Yurdakul & Naranjo (2020). \"Statistical properties of the population\n",
        "# stability index\". Journal of Risk Model Validation, 14(4), 89-100.\n",
        "# PSI = sum((p_i - q_i) * ln(p_i / q_i)); proven to follow a scaled chi-squared\n",
        "# distribution. Industry thresholds: <0.10 stable, 0.10-0.25 moderate, >=0.25 significant.\n",
        "\n",
    ],
}


def main():
    # Read notebook
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    cells = nb['cells']
    print(f"Notebook has {len(cells)} cells.")

    for cell_idx, ref_lines in REFERENCES.items():
        cell = cells[cell_idx]
        if cell['cell_type'] != 'code':
            print(f"WARNING: Cell {cell_idx} is '{cell['cell_type']}', not 'code'. Skipping.")
            continue

        # Show first line of existing content for verification
        first_line = cell['source'][0].strip() if cell['source'] else '(empty)'
        print(f"Cell {cell_idx}: inserting {len(ref_lines)} lines before: {first_line[:80]}")

        # Prepend reference lines
        cell['source'] = ref_lines + cell['source']

    # Safe-write: write to .tmp, validate, then rename
    tmp_path = NOTEBOOK_PATH + '.tmp'
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=True)

    # Validate the temp file is valid JSON
    with open(tmp_path, 'r', encoding='utf-8') as f:
        json.load(f)  # Will raise if invalid

    # Replace original with validated file
    shutil.move(tmp_path, NOTEBOOK_PATH)
    print(f"\nDone. References added to cells: {sorted(REFERENCES.keys())}")


if __name__ == '__main__':
    main()
