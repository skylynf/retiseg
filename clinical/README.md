# Clinical error review

Clinicians judge the regions where a segmentation model (the AI) and the dataset label disagree: what is there,
who is right, and what the error would change under ICDR and for the patient. Plan:
`docs/clinical/28_临床合作总计划.txt`; protocol for collaborators: `docs/clinical/29_protocol_en.md`; reader guide:
`docs/clinical/30_reader_guide_en.md`; meeting of 2026-10-29: `docs/clinical/31_会议纪要_20261029.txt`; trial
results and the formal design: `docs/clinical/32_试判结果与正式批次设计_20261103.txt`.
Readers see the lesion labels MA, HG, HEX and CWS; files and code keep the benchmark's class codes MA, HE, EX, SE.
Nothing here writes into the benchmark's own directories.

```
locate_landmarks.py     test set -> estimated fovea and disc per image + QC page for corrections (landmarks.json)
mine_discrepancies.py   prepared test set + probability maps of the model under review (+ optional model zoo, landmarks)
                        -> every disagreement / agreement region with its features, ETDRS zone included
                           (candidates.jsonl, images.jsonl, meta.json)
sample_review.py        frozen design (designs/*.yaml) -> two-stage sample, stratified by image and region strata
                        (selection.json = the key, with design weights)
build_pack.py           selection.json -> folder for the reader: index.html + pack.js + images/ + heat/ (--zip for e-mail)
reader_tool/index.html  offline single-file reader tool (ETDRS grid on the reader's fovea mark); answers saved in the
                        browser, exported as JSON
analyze_reviews.py      exports + selection.json (+ the pack) -> regions_long.csv, images_long.csv, report.md
schema/review_schema.json  the questionnaire (version inside; codes, options, display conditions); tool and analysis read it
```

Questionnaire 3.1.0 (docs/clinical/32, 33): no AI grade or referral question (the AI has no grade output); the band
of the closest hard exudate on the ETDRS grid (none / outer ring / inner ring, within 1 DD / central subfield, the
central 1 mm), recorded as a location and never as a DME diagnosis; the reader's own management for each eye; the
dataset label and the AI judged separately (so "both wrong" is two answers, never a missing option); ICDR thresholds
and HEX-band changes as separate options; optional IRMA / NV marks on the image.

## Formal batch, on the server (docs/ops/17 section 4 item 72)

```bash
# 0. fovea and disc estimates per test set (field angle: DDR, TJDR_std, MAPLES-DR 45; IDRiD 50; TJDR_uwf 133),
#    then open qc.html, click the misses, export corrections_<SET>.json and merge them
declare -A FIELD=([DDR]=45 [IDRiD]=50 [TJDR_std]=45 [TJDR_uwf]=133 [MAPLES-DR]=45)
for SET in DDR IDRiD TJDR_std TJDR_uwf MAPLES-DR; do
  python clinical/scripts/locate_landmarks.py estimate --set $SET --data dataset/prepared/$SET \
      --field-deg ${FIELD[$SET]} --out runs/CLINREV/landmarks/$SET
done
#    batch01: the corrections are committed (clinical/landmarks/batch01/, docs/clinical/34); TJDR_std needs none
for SET in DDR IDRiD TJDR_uwf MAPLES-DR; do
  python clinical/scripts/locate_landmarks.py apply --landmarks runs/CLINREV/landmarks/$SET/landmarks.json \
      --corrections clinical/landmarks/batch01/corrections_$SET.json
done
# 1. regions of RetiSeg-Ref with ETDRS zones, the 18 registered DDR-trained models as the zoo
for SET in DDR IDRiD TJDR_std TJDR_uwf MAPLES-DR; do
  python clinical/scripts/mine_discrepancies.py --set $SET --data dataset/prepared/$SET \
      --pred runs/ENS/REF/reference/$SET/prob_test --thresholds runs/release/retiseg_ref/thresholds.json \
      --model RetiSeg-Ref --grades auto --zoo-train DDR --landmarks runs/CLINREV/landmarks/$SET/landmarks.json \
      --out runs/CLINREV/mine/RetiSeg-Ref/$SET
done
# 2. population and eligible images per stratum, no draw (a stratum short of images gives the difference to
#    ddr_rest, images.shortfall_to, so the total stays 100), then freeze:
python clinical/scripts/sample_review.py --design clinical/designs/batch01.yaml --out runs/CLINREV/selection/batch01 --dry-run
#    commit designs/batch01.yaml; then the one draw of all 100 images, split into part 1 (20) and part 2 (80)
python clinical/scripts/sample_review.py --design clinical/designs/batch01.yaml --out runs/CLINREV/selection/batch01
# 3. the reader folder of part 1 and a zip for e-mail (images included: never into git or the handoff pack);
#    part 2 later from the same selection.json, with --part 2
python clinical/scripts/build_pack.py --selection runs/CLINREV/selection/batch01/selection.json --part 1 \
    --out runs/CLINREV/packs/batch01_part1 --title "RetiSeg clinical review - batch 1, part 1" --zip
```
Sampling (docs/clinical/33): with `reading_order: blind_first` every region stratum of an image is drawn once, with
no rejection, so inclusion probabilities are exact; the reader tool (1.1.0) asks step A for every region of an
image before one reveal, so overlapping crop windows cannot unblind anything. The trial designs keep the old
non-overlap rejection only so they reproduce; the stratified arm refuses to run without `blind_first`.
batch01 (frozen 2026-11-10) leaves MAPLES-DR out: the MESSIDOR terms prohibit copying and redistribution, and
build_pack.py refuses a pack holding MESSIDOR, FGADR, Retinal-Lesions or DiaRetDB1 images unless `--team-only`.
Every pack's READ_ME_FIRST.txt names the sources and licences of its images. The frozen design records the sha256 of
the mining outputs it was fixed on (`sets[].inputs`), and the sampler stops if they differ. The landmark estimates
only decide the strata; the analysis uses each reader's own marks.

## Fallback trial on the workstation

Kept for the record: the trial was read on the demo pack on 2026-11-02 (docs/clinical/32), so trial01.yaml and
trial01_workstation.yaml were never run. The workstation route below still works for a quick feedback pack:

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
python clinical/scripts/analyze_reviews.py --selection runs/CLINREV/selection/batch01/selection.json \
    --pack runs/CLINREV/packs/batch01_part1 [runs/CLINREV/packs/batch01_part2] \
    --reviews runs/CLINREV/reviews/batch01/*.json --out runs/CLINREV/analysis/batch01
```
Weights are the exact design weights scaled, per reader and image stratum, by drawn / finished images, so the same
command serves part 1 alone, both parts, or a partial read.
Exports of questionnaire 2 (the trial) are read too. `--pack` gives the DME implied by the label and by the AI and what
lies under each IRMA / NV mark; without it the implied DME comes from the estimated fovea.

## Try it

`designs/demo_idrid.yaml` built the pack of the trial (four IDRiD test images, M2MRF author checkpoint);
`designs/demo_idrid_strat.yaml` runs the formal design on the same images, in two parts of two images (its
header has the commands). Tests: `python -m pytest clinical/tests -q` (synthetic data; no dataset, weights or torch).
