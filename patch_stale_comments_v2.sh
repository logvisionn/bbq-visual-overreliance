#!/usr/bin/env bash
# Fix the two remaining stale "control: should ≈ 0" references in
# analyse_results.py that my earlier patch_stale_comments.sh missed.
#
# The C1 vs C2 comparison is NOT a null-hypothesis control — the data
# show a genuine −10.6 pp architecture-mode penalty on Qwen2-VL-7B.
# Calling it a "control" expected to vanish contradicts the findings
# in §5.2, §5.6, and §6.1 of the write-up.

set -euo pipefail
cd ~/bbq-visual-overreliance

echo "=== Before ==="
grep -nE 'control.*should.*0|Architecture mode effect.*control|controls.*architecture mode' \
    analyse_results.py || true

echo ""
echo "=== Applying fixes ==="

# Line 332 — inline comment
sed -i 's|# And C1 vs C2 (controls — should be ≈ 0 if architecture mode doesn.t matter)|# And C1 vs C2 (isolates the architecture-mode penalty from activating the vision encoder)|' \
    analyse_results.py

# Line 336 — description string in comparisons list
sed -i 's|"Architecture mode effect (control: should ≈ 0)"|"Architecture-mode penalty (vision encoder activation with neutral image)"|' \
    analyse_results.py

echo ""
echo "=== After ==="
grep -nE 'control.*should.*0|Architecture mode effect.*control|controls.*architecture mode' \
    analyse_results.py || echo "  (no more matches — all fixed)"

echo ""
echo "=== Verify script still parses ==="
python3 -m py_compile analyse_results.py && echo "  analyse_results.py: OK"

echo ""
echo "Done. Review 'git diff analyse_results.py' then commit."
