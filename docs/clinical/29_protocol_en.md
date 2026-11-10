# Where do lesion-segmentation AI models fail, and does it matter clinically?

**A clinician review of AI–label disagreements in diabetic retinopathy lesion segmentation**

Study protocol, version 2.2 (formal batch), 10 November 2026. Prepared by the RetiSeg team for our clinical collaborators. Version 1.0 (29 October) was revised after the trial read by Dr. Peranut Chotcomwongse on 2 November 2026 (version 2.0, 3 November) and checked again before the formal batch (version 2.1, 6 November). Version 2.2 fixes the sample: MAPLES-DR is not used, because the MESSIDOR terms do not allow its photographs to be copied to readers outside our team, and its 10 images go to DDR (section 7).

---

## 1. Summary

We have finished a standardised assessment of AI methods that outline four diabetic retinopathy (DR) lesions in color fundus photographs: microaneurysms (**MA**), hemorrhages (**HG**), hard exudates (**HEX**) and cotton wool spots (**CWS**). Nineteen models were retrained under one protocol and tested on eight test sets from seven public datasets. The benchmark tells us *how often* the AI disagrees with the published dataset labels, but not *who is right* or *whether it matters for patients*.

We are not asking you to draw lesions. We pick regions where our reference AI model and the dataset label disagree (plus some where they agree, as controls) and ask you, for each region:

1. what is actually there (before you see any outlines);
2. whether the dataset label and the AI are each right, and what is wrong with each (after the outlines are revealed);
3. which grading thresholds the error would cross, and whether, in your judgement, it would change your management of the patient.

The trial (4 images, 18 regions) showed that the questions and the tool work, and that regions drawn at random are almost all far from the fovea. The formal batch (100 images) therefore oversamples hard exudates near the fovea and suspected IRMA or new vessels, with exact design weights so that overall shares can still be estimated. It is read in two parts: **20 images first**, then the other 80.

## 2. What changed after the trial

| Trial finding | Change |
|---|---|
| None of 18 regions lay in the central subfield and only one within 1 disc diameter (DD); every error was judged clinically insignificant | Sampling stratified by location on the ETDRS grid (section 6) |
| The AI gives no DR grade, so "grade and referral from the AI lesions" could not be answered | Question removed; the hard-exudate location implied by the label and by the AI is measured from their outlines instead |
| "HEX at the foveal center" was too narrow and was read as a DME diagnosis | An ETDRS grid is drawn on the photograph, and the question records only where the closest hard exudate lies (central subfield, inner ring, outer ring); DME itself is not diagnosed from the photograph (section 5) |
| "Both wrong" was available but never used; one question mixed two judgements | The dataset label and the AI are judged in two separate questions |
| A faint IRMA, and a tortuous vessel the AI called a hemorrhage | IRMA and new vessels can be marked on the image; a region stratum targets suspected IRMA / NV |
| Revealing one region before answering the next exposed the outlines of the image | All regions of an image are answered blind first, then revealed together (section 8) |

## 3. Scope

- Only the four lesion types annotated in the public datasets (MA, HG, HEX, CWS) are judged region by region.
- IRMA, venous beading and new vessels (NVD/NVE), which decide severe NPDR and PDR, are not annotated in the datasets of this batch. You can tick and mark them; regions where the AI or the label calls a vessel-like structure a hemorrhage are oversampled; and the paper discusses them as a limitation of current benchmarks and a direction for future datasets.
- Hard exudates are located on the photograph; we do not diagnose diabetic macular edema (section 5).

## 4. Background: what the benchmark found

- Even the best models detect only about a quarter to a third of the smallest lesions, and roughly seven in ten microaneurysms the models report are not in the dataset label.
- Scores fall by a third to a half on images labeled in a different style (coarse outlines, region labels) or with a different field of view (ultra-widefield).
- On the one public dataset annotated independently by four experts, the experts agree least on microaneurysms.

Each of these findings mixes AI error with label error and with differences in lesion definitions. Only clinicians can separate them and say which errors matter.

## 5. Definitions used in this study

- **ICDR grades** (Wilkinson et al. 2003): no DR; mild NPDR (MA only); moderate NPDR (more than MA only, less than severe); severe NPDR (any of: more than 20 intraretinal hemorrhages in each of four quadrants, definite venous beading in two or more quadrants, prominent IRMA in one or more quadrants; no PDR); PDR (neovascularization and/or vitreous or preretinal hemorrhage). Of the four labeled lesion types, only the hemorrhage count can reach severe NPDR by itself; IRMA or new vessels outlined as hemorrhage can hide severe NPDR or PDR.
- **ETDRS grid**: centered on your foveal mark; central subfield 1 mm in diameter, inner ring 1–3 mm, outer ring 3–6 mm. Within 1 DD of the fovea = central subfield or inner ring, taking 1 DD = 1.5 mm as the ETDRS convention (real discs average about 1.8 mm, so "within 1 DD" judged against the disc itself reaches a little further). The scale takes your fovea-to-disc distance as 4.7 mm (the adult average is about 4.6–4.8 mm with an SD of about 7%), so in some eyes the circles may be 15–20% off; the grid is a guide.
- **Hard-exudate location**: the band of the grid in which the closest HEX lies. It is a location, not a diagnosis. For reference only: UK diabetic eye screening grades exudate within 1 DD of the center of the fovea as referable maculopathy (M1), and studies of AI screening have used hard exudate within 1 DD as a proxy for DME when stereo images are unavailable; the ETDRS counts hard exudate within 500 µm of the center as clinically significant macular edema only when the adjacent retina is thickened; in DRCR Retina Network trials, center-involved DME on OCT is central subfield thickness ≥320 µm (men) or ≥305 µm (women) on Spectralis. A single non-stereo photograph cannot show retinal thickening; stereo photographs or OCT can.
- **Management** is your own clinical judgement throughout. Grading thresholds and hard-exudate bands are recorded as descriptions; we do not equate them with a need to act.

## 6. Aims

1. **Who is right.** Of all AI–label disagreements, what share are AI errors, label errors, both, definition-dependent, or not resolvable from the photograph? By lesion type, size, dataset, DR grade and ETDRS location.
2. **What went wrong.** What the AI's false alarms really are (vessels, disc, reflexes, drusen, artifacts, IRMA or new vessels); where lesions are missed; what kinds of label error occur.
3. **Thresholds and management.** Which ICDR thresholds an error would cross, whether it would move the closest hard exudate across an ETDRS band, and whether, in your judgement, it would change management; where the rules and your judgement differ.
4. **Location.** Whether label omissions concentrate far from the fovea, and whether changes in management become rarer with distance on the ETDRS grid.
5. **Hard exudates.** Whether the hard exudates outlined by the dataset label and by the AI fall in the same ETDRS band as you place them.
6. **Direction.** What would make lesion AI clinically useful, and which lesion types future datasets should label.

## 7. Material and sample

**AI model under review.** RetiSeg-Ref, the reference model released with the benchmark (an average of two strong models trained on three public datasets with fine pixel labels). For each region we also record, without showing you, how many of 18 other published models make the same call.

**Photographs.** Official test images of public, de-identified datasets whose terms allow us to share them with you: IDRiD (CC BY 4.0), DDR (CC BY-NC-SA 4.0), and TJDR standard-field and ultra-widefield (released by its authors for direct download). Each folder lists its sources and licenses. No patient contact or identifiable information is involved. Images you read in the trial are excluded.

**Regions.** For every image and lesion type we overlay the dataset label and the AI output and split their union into connected regions: a lesion labeled but missed by the AI, an AI finding with no label, an extent disagreement, or an agreement (control).

**Location before reading.** For sampling only, the fovea and the optic disc of every image are estimated automatically and checked by eye. The analysis uses your own fovea and disc marks.

**Images (stage 1, 100).** Within each dataset, images are grouped in an ordered list of strata (an image belongs to the first that fits) and drawn at random within each. An image needs at least one disagreement to be eligible.

| Dataset (test images) | Strata, in order: images drawn |
|---|---|
| DDR (225; dataset grades available) | grade 3–4: 25; hard-exudate disagreement within 1 DD of the fovea: 25; grade 0–1: 10; the rest: 8, plus the images other strata lack (2 in this batch) |
| IDRiD (27, less the 4 read in the trial) | hard-exudate disagreement within 1 DD: 8; the rest: 4 (only 2 eligible) |
| TJDR standard field (55) | hard-exudate disagreement within 1 DD: 8; the rest: 3 |
| TJDR ultra-widefield (58) | hard-exudate disagreement within 1 DD: 5; the rest: 4 |

In all: DDR 70, IDRiD 10, TJDR standard field 11, TJDR ultra-widefield 9. A stratum with fewer eligible images than planned gives the difference to the last DDR stratum, so the total stays 100. These numbers were checked on the mined data before the design was fixed (10 November).

**Regions (stage 2, about 3–4 per image).** In each image, one region is drawn at random from each of: hard-exudate disagreements in the central subfield; in the inner ring; suspected IRMA or new vessels (a hemorrhage or microaneurysm disagreement that, in a DDR grade 3–4 image or a dataset without grades, is vessel-like or lies within 1.5 mm of the disc center); red-lesion disagreements within 1 DD; any other disagreement; agreements (controls). A region belongs to the first of these groups that fits. Each draw is made once, with no rejection or redraw, so every region's probability of being chosen is known exactly and overall shares are estimated with design weights.

**Two parts.** After the 100 images are drawn, 20 are set aside as part 1 in proportion to each image stratum, so part 1 is a random subsample of the batch; the other 80 form part 2. The design, the random seed and the analysis plan are committed before sampling.

## 8. What a reader does, and what is blinded

Everything happens in one offline web page that opens from the data folder by double-click (no installation, no internet). For each photograph:

| Step | What you see | Questions |
|---|---|---|
| 1. Grade the image | The photograph; the ETDRS grid once you mark the fovea (scaled from your disc mark, or the folder's estimate until then) | Image quality; fovea and disc (or "not visible"); ICDR grade (or "cannot grade"); band of the closest hard exudate (none on the grid, outer ring, inner ring, central subfield, cannot assess); optional: IRMA, NVD/NVE, venous beading, preretinal or vitreous hemorrhage, laser scars, with optional marks for IRMA and NV; your management for this eye from this photograph |
| 2. Step A, every region in turn | The photograph and one dashed circle at a time; no outlines | What is in the circle; confidence |
| 3. Reveal | One click after the last step A of the image | All step A answers of the image are locked (step 1 locked when you moved on to step A) |
| 4. Step B, every region in turn | Dataset label (solid) and AI (dashed) inside the square around the region; optional heat map; the ETDRS location of the circle's center | Dataset label in this region; AI in this region (each: correct, wrong extent, wrong lesion type, marks a non-lesion, misses a lesion, can't decide); if either is not correct: which thresholds the error would cross, and whether it would change your management; optional comment |
| 5. Whole image | All outlines and the grid; for DDR images, the dataset's own ICDR grade | Lesions missing from the dataset label; quality of the label and of the AI map; agreement with the dataset's grade where shown |

A short feedback page at the end asks which questions were useful, which options were missing or unclear, and whether the ETDRS grid and the hard-exudate question worked.

**Blinding, step by step.**

- *Step 1* shows only the photograph: no regions, outlines or heat maps, and no dataset name, image identifier or dataset grade. The camera, field of view and image quality may still reveal the source (ultra-widefield images are obvious). Your step 1 answers lock when you move on to step A, because the circles show where the label or the AI drew lesions. The progress bar shows how many regions the image has.
- *Step A* shows the photograph and the current region's circle. You are not told whether the dataset label, the AI or both drew a lesion there, which lesion type, which stratum the region came from, or the AI's probability. You do know that at least one of them drew one of the four lesion types inside the circle, and that this batch oversamples regions near the fovea and vessel-like regions. The circle's size and the zoom follow the size of the outlined area. All step A answers of an image are given before any outline of that image appears; you can change them until the reveal.
- *Across images*, earlier reveals let you learn the general style of the dataset label and of the AI; this cannot be hidden. Your image and region order is random, so we can check for order effects.
- *Step B and the whole-image step are not blinded*: you see which outline is the label and which is the AI, because that is what you are judging. The whole-image step also shows the dataset's own ICDR grade for DDR images, which identifies those images as DDR. Our primary results therefore use step A.
- Blinding depends on reading through the page: the data files in the folder also contain the outlines.
- Readers work independently. We will not discuss your part 1 answers or results with you until part 2 is finished, apart from problems with the tool or the questions.
- We, the analysts, hold the key and are not blinded; the design and analysis plan are fixed and committed before sampling.

## 9. Thresholds and management

**If the wrong outline were used for grading, which threshold would it cross?** (several may apply; "None" excludes the others), judged against your own grading of the image:

| Option | When it applies |
|---|---|
| None | ICDR grade and hard-exudate band unchanged |
| No DR ↔ mild NPDR | The only MA in the eye |
| No DR ↔ moderate NPDR | The only HG, HEX or CWS in an eye without MA |
| Mild ↔ moderate NPDR | The only HG, HEX or CWS in an MA-only eye |
| Moderate ↔ severe NPDR | Any 4-2-1 criterion, for example more than 20 intraretinal HG in each of four quadrants, or IRMA outlined as HG |
| NPDR ↔ PDR | For example new vessels outlined as HG |
| Closest HEX moves into or out of 1 DD | The inner-ring boundary, 1.5 mm from the fovea |
| Closest HEX moves into or out of the central subfield | The 0.5 mm boundary |

If both outlines are wrong, tick what either error would change. The grade options describe ICDR rules; many AI studies define referable DR as moderate NPDR or worse and/or referable DME (Gulshan et al. 2016), and UK screening refers at R2, R3 or M1. The hard-exudate options describe location only.

**If the wrong outline were used, would your management of this patient differ?** No / follow-up interval / referral or treatment / unsure. This is your own judgement, read against your management for the eye from step 1. Comparing it with the threshold question shows where a rule is crossed but practice would not change, and the reverse.

## 10. Analysis

- **Who is right (primary):** design-weighted shares of AI-only regions that you identify, blinded, as the lesion the AI reported (label omissions), and of labeled regions you confirm (true AI misses), by lesion type and ETDRS zone. After the reveal, the two outline questions give: both right, label wrong, AI wrong, both wrong, or undecided.
- **Thresholds and management:** weighted shares of each ICDR threshold, of hard-exudate band changes and of management changes, for all disagreements and separately within and beyond 1 DD; their cross-tabulations; unweighted results per region stratum.
- **Location:** each region's distance from your foveal mark in micrometers (closest point of the region, which can be nearer than the circle's center shown in step B; scale from your fovea-to-disc distance), grouped as central subfield (≤0.5 mm), inner ring (0.5–1.5 mm), outer ring (1.5–3 mm) and outside the grid, fixed before reading.
- **Hard exudates:** your band against the band of the closest hard exudate outlined by the dataset label and by the AI, measured from your foveal mark.
- **IRMA and new vessels:** what the four-class label and AI put under each of your IRMA / NV marks.
- **Weights:** each region's weight is the inverse of its exact probability of selection, scaled within each image stratum by the share of images you finished; this keeps estimates valid after part 1 alone, after both parts, or after a partial read. 95% intervals come from a bootstrap of images within image strata.
- **Secondary:** probability that the label is wrong given how many models disagree with it; lesion-level precision and recall after correcting label errors; agreement between readers (kappa, Gwet's AC1, weighted kappa for grades and bands); accuracy of the automatic fovea estimates against your marks.
- An interim analysis after part 1 checks the tool, the questions and the timing. The questions are fixed before part 1; between the parts only wording may change. If a substantive change were needed, part 1 would be reported separately.

## 11. Time commitment

| Activity | Time |
|---|---|
| Trial (done): 4 images, 18 regions | about 40 minutes of active reading |
| Part 1: 20 images, about 65–70 regions | projected 1.5–2 hours, can be split; part 1 will check this |
| Part 2: 80 images, about 260 regions | projected 6–8 hours, in sittings of 10–15 images |

Projections assume about 4–6 minutes per image; in the trial, images after the first took 5–9 minutes with an earlier version of the questions.

## 12. Outputs and authorship

The formal batch is planned for the benchmark paper (in preparation; target *Nature Communications*) as a section on what AI errors mean clinically, or as a companion paper on a clinical audit of DR lesion benchmarks. Participating clinicians are invited as co-authors, responsible for the clinical sections and interpretation; author order and contributions (CRediT) are agreed before submission.

## 13. Data handling and ethics

- Public, de-identified research datasets only; no patient contact; no new images.
- Images are shared only as the study folder, under each dataset's license; datasets whose license forbids redistribution are excluded. Please do not pass the folder on, and delete it when the study closes.
- Your answers stay on your computer until you export them; we keep them outside the public code repository until publication, linked to a reader code.
- Please tell us whether your institution requires an IRB determination for reading public de-identified images.

## 14. Timeline (proposed)

| When | What |
|---|---|
| 2 Nov | Trial read (done) |
| 3–6 Nov | Questions, tool, sampling and analysis revised (version 2.1) |
| 7–10 Nov | Fovea and disc estimates checked; licenses checked; design fixed (10 Nov) |
| Week of 9 Nov | Part 1 folder sent |
| About one week after | Part 1 returned; interim check |
| Then | Part 2 folder sent |

## References

- Early Treatment Diabetic Retinopathy Study Research Group. Photocoagulation for diabetic macular edema. ETDRS report number 1. *Arch Ophthalmol* 1985;103:1796–1806.
- Wilkinson CP et al. Proposed international clinical diabetic retinopathy and diabetic macular edema disease severity scales. *Ophthalmology* 2003;110:1677–1682.
- NHS Diabetic Eye Screening Programme. Retinal image grading criteria: features-based grading outcomes guidance (gov.uk).
- Gulshan V et al. Development and validation of a deep learning algorithm for detection of diabetic retinopathy in retinal fundus photographs. *JAMA* 2016;316:2402–2410.
- Baker CW et al. Effect of initial management with aflibercept vs laser photocoagulation vs observation on vision loss among patients with diabetic macular edema involving the center of the macula and good visual acuity. *JAMA* 2019;321:1880–1894 (DRCR Retina Network Protocol V).
- Jonas RA et al. Optic disc–fovea distance, axial length and parapapillary zones. The Beijing Eye Study 2011. *PLoS One* 2015;10:e0138701.
