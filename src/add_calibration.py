"""
Insert isotonic calibration cell into NB03 after the calibration curve (Cell 30).
Adds 1 markdown cell + 1 code cell as new Section 8.1b.
"""

import json
import shutil

NOTEBOOK_PATH = 'notebooks/03_model_training_evaluation.ipynb'

# New markdown cell: Section 8.1b header
markdown_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 8.1b Post-Hoc Calibration (Isotonic Regression)\n",
        "\n",
        "The calibration curve above shows the model **systematically overestimates default risk** — predicted probabilities are 3-5x higher than observed default rates. This is a direct consequence of `scale_pos_weight = 11.39`, which improves discrimination (AUC) at the cost of calibration.\n",
        "\n",
        "**Regulatory requirement:** FINMA Circular 2017/1, Basel IRB (CRR Art. 179), and the EU AI Act (Art. 15) all require calibrated PD estimates. Inflated probabilities would overstate Expected Loss, over-provision IFRS 9 Stage 2/3, and mislead customers under nDSG Art. 21.\n",
        "\n",
        "**Fix:** Isotonic regression — a non-parametric, monotonic mapping that preserves ranking (AUC unchanged) while correcting the probability scale. This is the recommended approach from Bequé et al. (2017), Section 5."
    ]
}

# New code cell: Isotonic calibration
code_cell = {
    "cell_type": "code",
    "metadata": {},
    "source": [
        "# =============================================================================\n",
        "# 8.1b POST-HOC CALIBRATION (ISOTONIC REGRESSION)\n",
        "# =============================================================================\n",
        "# Reference: Beque, Coussement, Gayler & Lessmann (2017). \"Approaches for credit\n",
        "# scorecard calibration: An empirical analysis\". Knowledge-Based Systems, 134, 213-227.\n",
        "# Isotonic regression recommended as the best-performing post-hoc calibrator\n",
        "# for credit scorecards (Section 5, Table 4).\n",
        "\n",
        "from sklearn.isotonic import IsotonicRegression\n",
        "\n",
        "# Save uncalibrated predictions for comparison\n",
        "y_pred_proba_uncalibrated = y_pred_proba.copy()\n",
        "\n",
        "# Step 1: Generate training predictions for calibrator fitting\n",
        "y_train_proba_uncal = model_baseline.predict_proba(X_train)[:, 1]\n",
        "\n",
        "# Step 2: Fit isotonic calibrator on training data\n",
        "calibrator = IsotonicRegression(out_of_bounds='clip')\n",
        "calibrator.fit(y_train_proba_uncal, y_train)\n",
        "\n",
        "# Step 3: Apply to test predictions\n",
        "y_pred_proba = calibrator.predict(y_pred_proba_uncalibrated)\n",
        "\n",
        "print(\"=\" * 60)\n",
        "print(\"POST-HOC CALIBRATION RESULTS\")\n",
        "print(\"=\" * 60)\n",
        "print(f\"  Calibrator:         Isotonic Regression\")\n",
        "print(f\"  Training samples:   {len(y_train):,}\")\n",
        "print(f\"\")\n",
        "print(f\"  Before calibration:\")\n",
        "print(f\"    Mean predicted PD:  {y_pred_proba_uncalibrated.mean():.4f}\")\n",
        "print(f\"    Median predicted PD: {np.median(y_pred_proba_uncalibrated):.4f}\")\n",
        "print(f\"    Std predicted PD:   {y_pred_proba_uncalibrated.std():.4f}\")\n",
        "print(f\"\")\n",
        "print(f\"  After calibration:\")\n",
        "print(f\"    Mean predicted PD:  {y_pred_proba.mean():.4f}\")\n",
        "print(f\"    Median predicted PD: {np.median(y_pred_proba):.4f}\")\n",
        "print(f\"    Std predicted PD:   {y_pred_proba.std():.4f}\")\n",
        "print(f\"\")\n",
        "print(f\"  Actual default rate:  {y_test.mean():.4f}\")\n",
        "print(f\"\")\n",
        "print(f\"  Mean PD vs actual default rate gap:\")\n",
        "print(f\"    Before: {abs(y_pred_proba_uncalibrated.mean() - y_test.mean()):.4f}\")\n",
        "print(f\"    After:  {abs(y_pred_proba.mean() - y_test.mean()):.4f}\")\n",
        "\n",
        "# Step 4: Before/after calibration curves\n",
        "prob_true_before, prob_pred_before = calibration_curve(\n",
        "    y_test, y_pred_proba_uncalibrated, n_bins=10, strategy='uniform')\n",
        "prob_true_after, prob_pred_after = calibration_curve(\n",
        "    y_test, y_pred_proba, n_bins=10, strategy='uniform')\n",
        "\n",
        "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))\n",
        "\n",
        "# Before\n",
        "ax1.plot(prob_pred_before, prob_true_before, 'bo-', linewidth=2, label='Uncalibrated')\n",
        "ax1.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Perfect')\n",
        "ax1.set_xlabel('Mean Predicted Probability')\n",
        "ax1.set_ylabel('Fraction of Positives')\n",
        "ax1.set_title('Before Calibration', fontweight='bold')\n",
        "ax1.legend()\n",
        "ax1.grid(True, alpha=0.3)\n",
        "\n",
        "# After\n",
        "ax2.plot(prob_pred_after, prob_true_after, 'go-', linewidth=2, label='Calibrated (Isotonic)')\n",
        "ax2.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Perfect')\n",
        "ax2.set_xlabel('Mean Predicted Probability')\n",
        "ax2.set_ylabel('Fraction of Positives')\n",
        "ax2.set_title('After Calibration', fontweight='bold')\n",
        "ax2.legend()\n",
        "ax2.grid(True, alpha=0.3)\n",
        "\n",
        "plt.suptitle('Calibration Fix: Isotonic Regression', fontsize=13, fontweight='bold', y=1.02)\n",
        "plt.tight_layout()\n",
        "plt.savefig(REPORTS_PATH / 'calibration_before_after.png', dpi=150, bbox_inches='tight')\n",
        "plt.show()\n",
        "\n",
        "# Step 5: Save calibrator\n",
        "joblib.dump(calibrator, MODELS_PATH / 'calibrator.pkl')\n",
        "print(f\"\\nCalibrator saved to models/calibrator.pkl\")\n",
        "print(f\"\\nAll downstream cells now use calibrated probabilities.\")\n",
        "print(f\"AUC-ROC is unchanged (isotonic preserves ranking).\")\n",
    ],
    "outputs": [],
    "execution_count": None
}

# Read notebook
with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Notebook has {len(nb['cells'])} cells.")

# Insert after Cell 30 (calibration curve), before Cell 31 (Brier header)
insert_idx = 31
nb['cells'].insert(insert_idx, markdown_cell)
nb['cells'].insert(insert_idx + 1, code_cell)

print(f"Inserted 2 cells at index {insert_idx}-{insert_idx+1}")
print(f"Notebook now has {len(nb['cells'])} cells.")

# Safe write
tmp_path = NOTEBOOK_PATH + '.tmp'
with open(tmp_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=True)
with open(tmp_path, 'r', encoding='utf-8') as f:
    json.load(f)
shutil.move(tmp_path, NOTEBOOK_PATH)
print("Done. Notebook saved.")
