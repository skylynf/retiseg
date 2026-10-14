"""E5 inference tiers (docs/design/15 sections 4-6 under the docs/paper/16 scope).

a  zero-shot: SAM and SAM 2.1 automatic masks on a registered grid, SAM 3 text prompts (amg.py, sam3_text.py)
b  oracle prompts from the test labels, an upper bound (prompts.py, prompted.py, same_info.py)

Shared: frozen.py (the protocol and its readiness checks), geometry.py (canvases and windows),
score_rules.py (masks to pixel scores), outputs.py (run directories, records, evaluation).
Probability maps are written with bench.common.io.write_prob and scored by bench.eval only.
"""
