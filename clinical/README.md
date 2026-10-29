# Clinical error review

Clinicians judge the regions where a segmentation model (the AI) and the dataset label disagree: what is there,
who is right, and what the error would change under ICDR and for the patient. Plan:
`docs/clinical/28_临床合作总计划.txt`; protocol for collaborators: `docs/clinical/29_protocol_en.md`; reader guide:
`docs/clinical/30_reader_guide_en.md`; meeting of 2026-10-29: `docs/clinical/31_会议纪要_20261029.txt`.
Readers see the lesion labels MA, HG, HEX and CWS; files and code keep the benchmark's class codes MA, HE, EX, SE.
Nothing here writes into the benchmark's own directories.

```
mine_discrepancies.py   prepared test set + probability maps of the model under review (+ optional model zoo)
                        -> every disagreement / agreement region with its features   (candidates.jsonl, images.jsonl, meta.json)
sample_review.py        frozen design (designs/*.yaml) -> two-stage, two-arm sample   (selection.json = the key)
build_pack.py           selection.json -> folder for the reader: index.html + pack.js + images/ + heat/ (--zip for e-mail)
reader_tool/index.html  offline single-file reader tool; answers saved in the browser, exported as JSON
analyze_reviews.py      exports + selection.json -> regions_long.csv, images_long.csv, report.md
schema/review_schema.json  the questionnaire (version inside; codes, options, display conditions); tool and analysis read it
```

## Trial (10-20 images), on the server (docs/ops/17 section 4 item 72)

```bash
# 1. regions of RetiSeg-Ref on each test set, with the 18 registered DDR-trained models as the zoo
for SET in DDR IDRiD TJDR_std TJDR_uwf; do
  python clinical/scripts/mine_discrepancies.py --set $SET --data dataset/prepared/$SET \
      --pred runs/ENS/REF/reference/$SET/prob_test --thresholds runs/release/retiseg_ref/thresholds.json \
      --model RetiSeg-Ref --grades auto --zoo-train DDR \
      --out runs/CLINREV/mine/RetiSeg-Ref/$SET
done
# 2. the frozen trial sample (seed and quotas in the design file)
python clinical/scripts/sample_review.py --design clinical/designs/trial01.yaml --out runs/CLINREV/selection/trial01
# 3. the reader folder and a zip for e-mail (images included: never into git or the handoff pack)
python clinical/scripts/build_pack.py --selection runs/CLINREV/selection/trial01/selection.json \
    --out clinical/packs/trial01 --title "RetiSeg clinical review - trial" --zip
```
The formal batch uses `designs/batch01.yaml` (draft, add MAPLES-DR to the loop if it stays in) after the trial feedback.

## Fallback trial on the workstation

If the RetiSeg-Ref maps are not ready when the trial has to go out, the workstation can build a trial from the M2MRF
author checkpoints it already has (feedback on the questions only; see the header of `designs/trial01_workstation.yaml`):

```bash
python clinical/scripts/mine_discrepancies.py --set DDR --data dataset/prepared/DDR --pred runs/E1_m2mrf/ddr_prob_test \
    --thresholds 0.5 --model "M2MRF author" --grades auto --out clinical/work/mine/M2MRF-author/DDR
python clinical/scripts/mine_discrepancies.py --set IDRiD --data dataset/prepared/IDRiD --pred runs/E1_m2mrf/prob_test \
    --thresholds 0.5 --model "M2MRF author" --out clinical/work/mine/M2MRF-author/IDRiD
python clinical/scripts/sample_review.py --design clinical/designs/trial01_workstation.yaml --out clinical/work/trial01w
python clinical/scripts/build_pack.py --selection clinical/work/trial01w/selection.json \
    --out clinical/packs/trial01w --title "RetiSeg clinical review - trial" --zip
```
`--grades auto` reads `runs/CLIN/grades.json` when present and otherwise the DDR release's own grade lists.

## After reading

```bash
python clinical/scripts/analyze_reviews.py --selection clinical/selections/trial01/selection.json \
    --reviews clinical/reviews/trial01/*.json --out clinical/work/trial01/analysis
```

## Try it

`clinical/packs/demo_idrid/index.html` is a demo built from five IDRiD test images and the M2MRF author checkpoint
(`designs/demo_idrid.yaml`). Tests: `python -m pytest clinical/tests -q` (synthetic data; no dataset, weights or torch).
