# Where do lesion-segmentation AI models fail, and does it matter clinically?

**A clinician review of AI–label disagreements in diabetic retinopathy lesion segmentation**

Study protocol, version 2.0 (formal batch), 3 November 2026. Prepared by the RetiSeg team for our clinical collaborators. Version 1.0 (29 October) was revised after the trial read by Dr. Peranut Chotcomwongse on 2 November 2026 and her comments.

---

## 1. Summary

We have finished a standardised assessment of AI methods that outline four diabetic retinopathy (DR) lesions in colour fundus photographs: microaneurysms (**MA**), hemorrhages (**HG**), hard exudates (**HEX**) and cotton wool spots (**CWS**). Nineteen models were retrained under one protocol and tested on eight test sets from seven public datasets. The benchmark tells us *how often* the AI disagrees with the published dataset labels, but not *who is right* or *whether it matters for patients*.

We are not asking you to draw lesions. We pick regions where our reference AI model and the dataset label disagree (plus some where they agree, as controls) and ask you, for each region:

1. what is actually there (before you see any outlines);
2. whether the dataset label and the AI are each right, and what is wrong with each (after the outlines are revealed);
3. what the error would change under ICDR and for DME, and whether it would change what you do for the patient.

The trial (4 images, 18 regions) showed that the questions and the tool work, and that regions drawn at random are almost all far from the fovea, where errors rarely matter. The formal batch (about 100 images) therefore oversamples hard exudates near the fovea and suspected IRMA or new vessels, with design weights so that overall shares can still be estimated.

## 2. What changed after the trial

| Trial finding | Change in version 2.0 |
|---|---|
| None of 18 regions lay in the central subfield and only one within 1 disc diameter (DD); every error was judged clinically insignificant | Sampling stratified by location on the ETDRS grid (section 6) |
| The AI gives no DR grade, so "grade and referral from the AI lesions" could not be answered | Question removed; DME implied by the label and by the AI is measured from their outlines instead |
| "HEX at the foveal center" is too narrow a photographic surrogate for center-involving DME, which is defined by OCT central subfield thickness (≥320 µm in men, ≥305 µm in women) | DME is judged on an ETDRS grid drawn on the photograph: closest HEX in the central subfield (1 mm), inner ring (within 1 DD) or outer ring |
| "Both wrong" was available but never used; one question mixed two judgements | The dataset label and the AI are judged in two separate questions |
| A faint IRMA, and a tortuous vessel the AI called a hemorrhage | IRMA and new vessels can be marked on the image; a region stratum targets suspected IRMA / NV |

## 3. Scope

- Only the four lesion types annotated in the public datasets (MA, HG, HEX, CWS) are judged region by region.
- IRMA, venous beading and new vessels (NVD/NVE), which decide severe NPDR and PDR, are not annotated in most of these datasets (MAPLES-DR annotates new vessels). You can tick and mark them; regions where the AI or the label calls a vessel-like structure a hemorrhage are oversampled; and the paper discusses them as a limitation of current benchmarks and a direction for future datasets.
- DME is judged on the photograph as a surrogate (hard exudates relative to the fovea on the ETDRS grid). We do not claim center-involving DME in the OCT sense.

## 4. Background: what the benchmark found

- Even the best models detect only about a quarter to a third of the smallest lesions, and roughly seven in ten microaneurysms the models report are not in the dataset label.
- Scores fall by a third to a half on images labelled in a different style (coarse outlines, region labels) or with a different field of view (ultra-widefield).
- On the one public dataset annotated independently by four experts, the experts agree least on microaneurysms.

Each of these findings mixes AI error with label error and with differences in lesion definitions. Only clinicians can separate them and say which errors matter.

## 5. Aims

1. **Who is right.** Of all AI–label disagreements, what share are AI errors, label errors, both, definition-dependent, or not resolvable from the photograph? By lesion type, size, dataset, DR grade and ETDRS location.
2. **What went wrong.** What the AI's false alarms really are (vessels, disc, reflexes, drusen, artifacts, IRMA or new vessels); where lesions are missed; what kinds of label error occur.
3. **Clinical consequence.** Which ICDR and DME thresholds an error would cross, and whether it would change management; where the two differ.
4. **Location.** Whether label omissions concentrate far from the fovea, and whether clinical relevance falls with distance on the ETDRS grid.
5. **DME.** Whether the hard exudates outlined by the dataset label and by the AI place the eye in the same ETDRS zone as you do.
6. **Direction.** What would make lesion AI clinically useful, and which lesion types future datasets should label.

## 6. Material and sample

**AI model under review.** RetiSeg-Ref, the reference model released with the benchmark (an average of two strong models trained on three public datasets with fine pixel labels). For each region we also record, without showing you, how many of 18 other published models make the same call.

**Photographs.** Official test images of public, de-identified datasets whose terms allow you to view them: IDRiD, DDR, TJDR standard-field and ultra-widefield, and MAPLES-DR if the MESSIDOR terms allow. No patient contact or identifiable information is involved. Images you read in the trial are excluded.

**Regions.** For every image and lesion type we overlay the dataset label and the AI output and split their union into connected regions: a lesion labelled but missed by the AI, an AI finding with no label, an extent disagreement, or an agreement (control). You are never told which.

**Location before reading.** For sampling only, the fovea and the optic disc of every image are estimated automatically and checked by eye; the ETDRS scale takes the fovea-to-disc distance as 4.5 mm (1 DD = 1.5 mm). The analysis uses your own fovea and disc marks.

**Images (stage 1, about 100).** Within each dataset, images are grouped in an ordered list of strata and drawn at random within each: DDR grade 3–4 (where IRMA, new vessels and the 4-2-1 hemorrhage rule matter), images with a hard-exudate disagreement within 1 DD of the fovea, DDR grade 0–1 (where a single lesion decides the grade), and the rest.

**Regions (stage 2, about 3–4 per image).** In each image, at most one region from each of: a hard-exudate disagreement in the central subfield; one in the inner ring; a suspected IRMA or new vessel (a hemorrhage or microaneurysm disagreement that is vessel-like, lies on or next to the disc, or overlaps labelled new vessels); a red-lesion disagreement within 1 DD; any other disagreement; and an agreement (control).

Every inclusion probability is recorded, so overall shares are estimated with design weights; the strata near the fovea are read far more often than their share. Regions in the same photograph never overlap, so revealing one region's outlines never shows the answer for another. The design and random seed are fixed and committed before sampling.

## 7. What a reader does

Everything happens in one offline web page that opens from the data folder by double-click (no installation, no internet). For each photograph:

| Step | What you see | Questions |
|---|---|---|
| Grade the image | The photograph; the ETDRS grid once you mark the fovea and disc | Image quality; fovea and disc (or "not visible"); ICDR grade; DME (closest HEX: none on the grid, outer ring, inner ring, central subfield); optional: IRMA, NVD/NVE, venous beading, preretinal or vitreous hemorrhage, laser scars, with optional marks for IRMA and NV |
| Region, step A | A dashed circle, no outlines | What is in the circle; confidence |
| Reveal | Dataset label (solid) and AI (dashed) inside the region's window; optional heat map; the region's ETDRS location | Your step-A answer is locked |
| Region, step B | Both outlines | Dataset label in this region; AI in this region (each: correct, wrong extent, wrong lesion type, marks a non-lesion, misses a lesion, can't decide); if either is not correct: what it would change under ICDR / DME, and whether it would change what you do; optional comment |
| Whole image | All outlines and the grid | Lesions missing from the dataset label; quality of the label and of the AI map; agreement with the dataset's own grade where available |

A short feedback page at the end asks which questions were useful, which options were missing or unclear, and whether the ETDRS grid and the DME question worked.

## 8. Clinical consequence

**If the wrong outline were used to grade this eye, what would change?** (several may apply; "Nothing" excludes the others), judged against your own grading of the image:

| Option | When it applies | Level |
|---|---|---|
| Nothing | Grade and DME unchanged (for example, many lesions of the same type elsewhere) | 0 |
| No DR ↔ mild NPDR | The only MA in the eye | 1 minor |
| Mild ↔ moderate NPDR | The only HG, HEX or CWS in an MA-only eye; this also crosses the referral threshold | 2 moderate |
| DME: HEX within 1 DD | The closest HEX moves into or out of the inner ring (referable DME on photographs) | 2 moderate |
| DME: HEX in the central subfield | The closest HEX moves into or out of the central 1 mm (the photographic sign closest to center-involving DME) | 3 major |
| Moderate ↔ severe NPDR | The hemorrhage criterion of the 4-2-1 rule (more than 20 HG in each of four quadrants); of the four lesion types, only HG can reach severe NPDR | 3 major |

**Would it change what you do for this patient?** No (for example, a small lesion far from the fovea) / follow-up interval / referral or treatment / unsure.

Comparing the two shows where a rule-based threshold is crossed but clinical practice would not change.

## 9. Analysis

- **Who is right (primary):** design-weighted shares of AI-only regions that you identify as the lesion the AI reported (label omissions), and of labelled regions you confirm (true AI misses), by lesion type and ETDRS zone, with 95% intervals from a bootstrap over images. Primary results use the blinded step-A answers. After the reveal, the two outline questions give: both right, label wrong, AI wrong, both wrong, or undecided.
- **Clinical consequence:** weighted shares of each consequence level and of management change, for all disagreements and separately within and beyond 1 DD of the fovea; their cross-tabulation; unweighted results per region stratum.
- **Location:** each region's distance from your foveal mark in micrometres (closest point of the region; scale from your fovea-to-disc distance), grouped as central subfield (≤0.5 mm), inner ring (0.5–1.5 mm), outer ring (1.5–3 mm) and outside the grid, fixed before reading.
- **DME:** your DME zone against the zone of the closest hard exudate outlined by the dataset label and by the AI, measured from your foveal mark.
- **IRMA and new vessels:** what the four-class label and AI put under each of your IRMA / NV marks.
- **Secondary:** probability that the label is wrong given how many models disagree with it; lesion-level precision and recall after correcting label errors; agreement between readers (kappa, Gwet's AC1, weighted kappa for grades and zones); accuracy of the automatic fovea estimates against your marks.
- Each reader's image order is random, so a partial read is a random subsample: estimates remain valid with wider intervals.

## 10. Time commitment

| Activity | Time |
|---|---|
| Trial (done): 4 images, 18 regions | about 40 minutes of active reading |
| Formal batch, per reader: about 100 images, about 330 regions | about 7–10 h (about 4–6 min per image), in sittings of 20–25 images |

## 11. Outputs and authorship

The formal batch is planned for the benchmark paper (in preparation; target *Nature Communications*) as a section on what AI errors mean clinically, or as a companion paper on a clinical audit of DR lesion benchmarks. Participating clinicians are invited as co-authors, responsible for the clinical sections and interpretation; author order and contributions (CRediT) are agreed before submission.

## 12. Data handling and ethics

- Public, de-identified research datasets only; no patient contact; no new images.
- Images are shared only as the study folder, under each dataset's licence; datasets whose licence forbids redistribution are excluded. Please do not pass the folder on, and delete it when the study closes.
- Your answers stay on your computer until you export them; we keep them outside the public code repository until publication, linked to a reader code.
- Please tell us whether your institution requires an IRB determination for reading public de-identified images.

## 13. Timeline (proposed)

| When | What |
|---|---|
| 2 Nov | Trial read (done) |
| 3 Nov | Questions, tool and sampling revised (this version) |
| 4–7 Nov | Licence checks; fovea and disc estimates checked; design fixed; readers confirmed |
| 8–10 Nov | Study folder sent |
| From 10 Nov | Formal batch |
