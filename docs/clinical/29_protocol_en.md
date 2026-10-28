# Where do lesion-segmentation models fail, and does it matter clinically?

**A clinician review of model–annotation disagreements in diabetic retinopathy lesion segmentation**

Study protocol, version 0.9 (pilot), 28 October 2026. Prepared by the RetiSeg team for our clinical collaborators.

---

## 1. Summary

We have finished a large, standardised assessment of methods that segment four diabetic retinopathy (DR) lesions in colour fundus photographs: microaneurysms (MA), haemorrhages (HE), hard exudates (EX) and cotton-wool spots (SE). Nineteen models were retrained under one protocol and tested on eight test sets from seven public datasets. The benchmark tells us *how often* models disagree with the published annotations, but not *why* they disagree or *whether it matters for patients*.

We are not asking you to draw lesions from scratch. Instead, we pick regions where our reference model and the dataset annotation disagree (plus some where they agree, as controls) and ask you to judge, for each region:

1. what is actually there (before you see any outlines),
2. who is wrong and why (after the outlines are revealed),
3. how much the error would change grading or referral, and whether it is decisive,
4. what would most reduce this kind of error.

The same reading audits the public datasets, audits the models, and gives evidence on where the field should go next. We propose to start with a **pilot of about 100 regions from 24 photographs**, which takes each reader roughly **2 to 2.5 hours**, split into two sittings if you prefer.

## 2. Background: what the benchmark found

- Pixel-level scores are modest and vary widely by lesion. Even the best models detect only about a quarter to a third of the smallest lesions, and roughly seven in ten microaneurysms the models report are not in the annotation.
- Scores fall by a third to a half when a model meets photographs annotated in a different style (coarse outlines, region-level labels) or a different field of view (ultra-widefield).
- On the one public dataset annotated independently by four experts, the experts agree least on microaneurysms.
- Scores differ greatly between mild images (grades 0–1) and more advanced ones (grades 2–4).

Each of these findings mixes model error with annotation error and with differences in how lesions are defined. Only clinicians can separate them, and only clinicians can say which errors would change a patient's management.

## 3. Aims

1. **Attribution.** Of all disagreements between model and annotation, what share are model errors, annotation errors, both, definition-dependent, or not resolvable from the photograph? By lesion type, lesion size, dataset and DR grade.
2. **Mechanisms.** Which look-alikes cause false alarms (vessel crossings, peripapillary changes, light reflexes, drusen, pigment, artifacts); in which situations are lesions missed (small, faint, crowded, poor local quality, periphery); which annotation errors occur (omissions, false labels, wrong class, outlines that enclose several lesions, dataset conventions).
3. **Clinical consequence.** How often would an error change the ICDR grade, the DME assessment or the referral decision, and how often is the lesion at stake decisive on its own?
4. **Metrics versus medicine.** Are the errors that the benchmark metrics penalise the ones clinicians care about? What are the models' lesion-level precision and recall once annotation errors are corrected?
5. **Direction.** What would most reduce these errors, which lesion classes should a benchmark add, and where would pixel-level lesion maps be useful in practice?

## 4. Material

**Model under review.** RetiSeg-Ref, the reference model released with the benchmark: an average of two strong segmentation models trained on three public datasets with fine pixel annotations. Its outlines are what any user would get from the released model. For each region we also record (without showing you) how many of 18 other published models make the same call, so that disagreements shared by nearly all models can be compared with your judgement.

**Photographs.** Official test images of public, de-identified datasets whose terms allow you to view them for this study: IDRiD (India), DDR (China), TJDR standard-field and ultra-widefield (China), and, if the MESSIDOR terms allow, MAPLES-DR (France). No patient contact or identifiable information is involved.

**Regions.** For every image and lesion type we overlay the dataset annotation and the model's output and split their union into connected regions. Each region is one of: a lesion annotated but missed by the model, a model finding with no annotation, an extent disagreement, or an agreement (control). You are never told which.

## 5. Sampling for the pilot

- 24 photographs drawn at random within strata: DDR grade 0–1 (4), DDR grade 2–4 (8), IDRiD (4), TJDR standard-field (3), TJDR ultra-widefield (3), MAPLES-DR (2).
- In each photograph about 3 regions drawn at random (one missed lesion, one model-only finding, one agreement), about 70 in total. These carry sampling weights, so the shares we report generalise to all disagreements of the model.
- 28 further regions chosen on purpose to study mechanisms: confident false alarms, confidently missed larger lesions, class confusions, extent disagreements, and false alarms shared by nearly all models.
- Regions in the same photograph never overlap, so revealing one region's outlines never shows you the answer for another.
- The design and random seed are fixed and recorded before any reading starts.

## 6. What a reader does

Everything happens in one offline web page that opens from the data folder by double-click (no installation, no internet). For each photograph:

| Step | What you see | What you record |
|---|---|---|
| Image assessment | The photograph only | Image quality; ICDR DR grade; DME (photographic surrogate); other findings (IRMA, NV, venous beading, pre-retinal/vitreous haemorrhage, laser scars, drusen, …); optional clicks on the fovea and the disc |
| Region, step A | A dashed circle, no outlines | What is in the circle (lesion types, normal structures, artifacts, or cannot determine); confidence; more than one lesion? |
| Reveal | Dataset annotation (solid) and model (dashed) inside the region's window; optional heat map | Your step-A answer is locked |
| Region, step B | Both outlines | Main explanation; what is wrong with the annotation and/or the model; difficulty for a human grader; clinical impact (0–3); decisive on its own?; decisions affected; what would reduce this error; free text "what went wrong here?" |
| Image summary | All outlines of the image | Would the model's lesions alone give the same grade and the same referral decision; overall quality of the annotation and of the model's map; agreement with the dataset's own grade where available |

**Clinical impact scale** (for the error in this region, in the context of this image):
0 none; 1 minor (lesion count or extent only); 2 moderate (changes DR grade or DME category, not referral); 3 major (changes the referral decision or its urgency). Separately: *decisive* means this single lesion by itself decides a grading threshold, e.g. the only microaneurysm (grade 0 vs 1), an exudate at the foveal centre, new vessels.

A short debrief at the end asks about the cost of each kind of miss and false alarm, where lesion maps would help, and which lesion classes a benchmark should add.

The page saves every answer automatically on your computer. At the end of each sitting you click *Export results* and send us the file. A reader guide with definitions and examples comes with the data folder.

## 7. Bias control

- Image order and region order are shuffled differently for each reader.
- You decide what is in the circle before any outline is shown; that answer is locked when you reveal. Our primary results use these blinded answers.
- Agreement regions are mixed in as controls, so we can measure any tendency to see a problem wherever a circle is drawn.
- You do not see model probabilities, model consensus or dataset names.
- Readers work independently until the pilot debrief meeting.

## 8. Time commitment

| Activity | Time per reader |
|---|---|
| Kick-off video call: protocol, tool demonstration, five practice regions, and writing down referral rules before seeing any results | 30–40 min |
| Pilot reading: about 100 regions in 24 photographs | 2–2.5 h (two sittings suggested) |
| Pilot debrief meeting | 30–45 min |
| Main study (size decided after the pilot) | to be agreed; typical options 4.5–10 h |

Please tell us if "about 100 annotations" meant 100 photographs rather than 100 regions; that would take roughly four times longer, and we would adapt the design.

## 9. What the pilot must show

The pilot tests feasibility before the main study. We will proceed unchanged if: median time per region is at most 60 s; "other lesion" and "cannot determine" together are below 15% of step-A answers; agreement between readers on the main explanation reaches kappa 0.4 (or AC1 0.5); each reader completes at least 90% of regions; and no tool problem blocks reading. Otherwise we revise wording, answer options or quotas (never the blinding or the primary endpoints) and record the change.

## 10. Analysis

- **Primary:** design-weighted shares of (a) model-only findings that you identify as the lesion the model reported (annotation omissions) and (b) annotated lesions you confirm (true model misses), by lesion type, with 95% intervals from a bootstrap over images; the distribution of the stated main explanation; the share of errors with impact ≥ 2 and = 3, and of decisive lesions.
- **Secondary:** mechanism frequencies; the probability that the annotation is wrong as a function of how many models disagree with it; lesion-level precision and recall after correcting for annotation errors; inter-reader agreement for every field (percent agreement, Cohen's and Fleiss' kappa, Gwet's AC1, weighted kappa for ordinal fields); reader grade versus dataset grade; agreement of model-implied grade and referral with yours; votes on remedies; thematic coding of free text.
- Each reader is reported separately and then combined (majority of three, or agreement of two); ties are settled at the debrief meeting and recorded alongside the original answers.

## 11. Outputs and authorship

The pilot is planned to enter the benchmark paper (in preparation; target *Nature Communications*) as a section on what model errors mean clinically. The full study is planned as a companion paper on a clinical audit of DR lesion benchmarks. Participating clinicians are invited as co-authors, with responsibility for the clinical sections and the clinical interpretation; we will agree author order and contributions (CRediT roles) before submission.

## 12. Data handling and ethics

- Public, de-identified research datasets only; no patient contact; no new images.
- Images are shared only as the study folder, for this study, under each dataset's licence; datasets whose licence forbids redistribution are excluded. Please do not pass the folder on, and delete it when the study closes.
- Your answers are stored on your computer until you export them and are kept by the study team outside the public code repository until publication. They are linked to a reader code, not your name.
- Please let us know whether your institution requires an IRB determination (for example "not human subjects research") for reading public de-identified images.

## 13. Timeline (proposed)

| When | What |
|---|---|
| By 1 Nov | You review this protocol; we confirm dataset terms |
| 2–5 Nov | We build and check the pilot folder |
| Around 6 Nov | Kick-off call; folder sent |
| 6–15 Nov | Pilot reading |
| 16–20 Nov | Pilot analysis and debrief meeting; main-study size agreed |
| From 21 Nov | Main study |

## 14. Questions for you

1. How many readers can take part, and roughly how much time can each give in November?
2. Did "about 100 annotations" mean 100 regions (as proposed) or 100 photographs?
3. Are the answer options in section 6 complete and in the terms you use? Anything you would add, merge or rename?
4. Is the 0–3 clinical-impact scale with a separate "decisive" flag the right way to capture consequence?
5. Do you need an IRB determination, and are there constraints on receiving the image folder?
