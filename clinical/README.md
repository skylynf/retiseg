# Clinical error review

Clinicians judge the regions where a segmentation model and the dataset annotation disagree: what is there,
who is wrong, why, and how much it matters for grading and referral. The plan is
`docs/clinical/28_临床合作总计划.txt` (protocol for collaborators: `docs/clinical/29_protocol_en.md`; reader guide:
`docs/clinical/30_reader_guide_en.md`). Nothing here writes into the benchmark's own directories.

```
mine_discrepancies.py   prepared test set + probability maps of the model under review (+ optional model zoo)
                        -> every disagreement / agreement region with its features   (candidates.jsonl, images.jsonl, meta.json)
sample_review.py        frozen design (designs/*.yaml) -> two-stage, two-arm sample   (selection.json = the key)
build_pack.py           selection.json -> folder for the reader: index.html + pack.js + images/ + heat/
reader_tool/index.html  offline single-file reader tool; answers saved in the browser, exported as JSON
analyze_reviews.py      exports + selection.json -> regions_long.csv, images_long.csv, report.md
schema/review_schema_v1.json  the questionnaire (codes, options, display conditions); the tool and the analysis both read it
```

## Pilot on the server (docs/ops/17 section 4 item 72)

```bash
# 1. regions of RetiSeg-Ref on each test set, with the 18 registered DDR-trained models as the zoo
for SET in DDR IDRiD TJDR_std TJDR_uwf MAPLES-DR; do
  python clinical/scripts/mine_discrepancies.py --set $SET --data dataset/prepared/$SET \
      --pred runs/ENS/REF/reference/$SET/prob_test --thresholds runs/release/retiseg_ref/thresholds.json \
      --model RetiSeg-Ref --grades runs/CLIN/grades.json --zoo-train DDR \
      --out runs/CLINREV/mine/RetiSeg-Ref/$SET
done
# 2. the frozen pilot sample (seed and quotas in the design file)
python clinical/scripts/sample_review.py --design clinical/designs/pilot01.yaml --out runs/CLINREV/selection/pilot01
# 3. the reader folder (images included: never into git or the handoff pack)
python clinical/scripts/build_pack.py --selection runs/CLINREV/selection/pilot01/selection.json \
    --out clinical/packs/pilot01 --title "RetiSeg clinical review - pilot"
```

## After reading (workstation)

```bash
python clinical/scripts/analyze_reviews.py --selection clinical/selections/pilot01/selection.json \
    --reviews clinical/reviews/pilot01/*.json --out clinical/work/pilot01/analysis
```

## Try it

`clinical/designs/demo_idrid.yaml` builds a demo from five IDRiD test images and any model's `prob_test`
(the demo used the M2MRF author checkpoint, `runs/E1_m2mrf`). Open `clinical/packs/demo_idrid/index.html`.

Tests: `python -m pytest clinical/tests -q` (synthetic data; no dataset, weights or torch).
