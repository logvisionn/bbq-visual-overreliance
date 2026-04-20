#!/usr/bin/env bash
# Patch stale "C1 ≈ C2 sanity check" framing in code comments.
# These comments were written pre-experiment when C1≈C2 was the working
# hypothesis. The actual results rejected C1≈C2 (7B: −10.6 pp), so the
# comments now misrepresent the interpretation.

set -euo pipefail
cd ~/bbq-visual-overreliance

echo "=== Before ==="
grep -nE "C1 ≈ C2|C1≈C2|sanity.check|SANITY CHECK" \
    run_full_experiment.py analyse_results.py || true

echo ""
echo "=== Applying fixes ==="

# run_full_experiment.py:14 — docstring framing
sed -i 's|This empirically measures whether C1 ≈ C2 (confirms that any C3 drop is|This empirically measures the C1→C2 gap (the "architecture-mode penalty"): any|' \
    run_full_experiment.py

# analyse_results.py:28 — top-of-file doc comment
sed -i 's|c1_vs_c2_comparison.csv        -- empirical check: does C1 ≈ C2?|c1_vs_c2_comparison.csv        -- C1 vs C2 comparison (architecture-mode penalty)|' \
    analyse_results.py

# analyse_results.py:410 — section log message
sed -i 's|── Checking whether C1 ≈ C2 (architecture-mode sanity check) ──|── Measuring C1 vs C2 gap (architecture-mode penalty) ──|' \
    analyse_results.py

# analyse_results.py:435 — interpretation string
sed -i 's|"C1≈C2 confirms text reasoning unchanged by vision encoder"|"C1 vs C2 quantifies the architecture-mode penalty: accuracy change from activating the vision encoder with a neutral image"|' \
    analyse_results.py

# analyse_results.py:489 — section header
sed -i 's|C1 vs C2 SANITY CHECK|C1 vs C2 — ARCHITECTURE-MODE PENALTY|' \
    analyse_results.py

echo ""
echo "=== After ==="
grep -nE "C1 ≈ C2|C1≈C2|sanity.check|SANITY CHECK" \
    run_full_experiment.py analyse_results.py || echo "  (no more matches — all fixed)"

echo ""
echo "=== Verify scripts still parse ==="
python3 -m py_compile run_full_experiment.py && echo "  run_full_experiment.py: OK"
python3 -m py_compile analyse_results.py     && echo "  analyse_results.py: OK"

echo ""
echo "Done. Review 'git diff' then commit."
