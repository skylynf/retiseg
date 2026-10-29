# Where do lesion-segmentation AI models fail, and does it matter clinically?

**A clinician review of AI–label disagreements in diabetic retinopathy lesion segmentation**

Study protocol, version 1.0 (trial), 29 October 2026. Prepared by the RetiSeg team for our clinical collaborators. Revised after the discussion with Dr. Peranut Chotcomwongse on 29 October 2026.

---

## 1. Summary

We have finished a standardised assessment of AI methods that outline four diabetic retinopathy (DR) lesions in colour fundus photographs: microaneurysms (**MA**), hemorrhages (**HG**), hard exudates (**HEX**) and cotton wool spots (**CWS**). Nineteen models were retrained under one protocol and tested on eight test sets from seven public datasets. The benchmark tells us *how often* the AI disagrees with the published dataset labels, but not *who is right* or *whether it matters for patients*.

We are not asking you to draw lesions. We pick regions where our reference AI model and the dataset label disagree (plus some where they agree, as controls) and ask you, for each region:

1. what is actually there (before you see any outlines);
2. who is right, and what went wrong (after the outlines are revealed);
3. what the error would change under ICDR, and whether it would change what you do for the patient.

We start with a **trial of 15 images (about 55 regions, about one hour)** to test the questions and the tool. The formal batch (around 100 images) is fixed after your feedback.

## 2. Scope

- Only the four lesion types annotated in the public datasets (MA, HG, HEX, CWS) are judged.
- IRMA, venous beading and new vessels (NVD/NVE), which decide severe NPDR and PDR, are **not annotated** in any of these datasets. You can record them as optional findings, and they are discussed in the paper as a limitation of current benchmarks and a direction for future datasets.
- Small lesions far from the fovea rarely change management, and human labels often miss them. We test both points explicitly (section 9).

## 3. Background: what the benchmark found

- Even the best models detect only about a quarter to a third of the smallest lesions, and roughly seven in ten microaneurysms the models report are not in the dataset label.
- Scores fall by a third to a half on images labelled in a different style (coarse outlines, region labels) or with a different field of view (ultra-widefield).
- On the one public dataset annotated independently by four experts, the experts agree least on microaneurysms.

Each of these findings mixes AI error with label error and with differences in lesion definitions. Only clinicians can separate them and say which errors matter.

## 4. Aims

1. **Who is right.** Of all AI–label disagreements, what share are AI errors, label errors, both, definition-dependent, or not resolvable from the photograph? By lesion type, size, dataset, DR grade and distance from the fovea.
2. **What went wrong.** What the AI's false alarms really are (vessels, disc, reflexes, drusen, artifacts); where lesions are missed; what kinds of label error occur.
3. **Clinical consequence.** Which ICDR thresholds an error would cross, and whether it would change management; where the two differ.
4. **Location.** Whether label omissions concentrate far from the fovea, and whether clinical relevance falls with distance.
5. **Direction.** What would make lesion AI clinically useful, and which lesion types future datasets should label.

## 5. Material

**AI model under review.** RetiSeg-Ref, the reference model released with the benchmark (an average of two strong models trained on three public datasets with fine pixel labels). For each region we also record, without showing you, how many of 18 other published models make the same call.

**Photographs.** Official test images of public, de-identified datasets whose terms allow you to view them: IDRiD, DDR, and TJDR standard-field and ultra-widefield. No patient contact or identifiable information is involved.

**Regions.** For every image and lesion type we overlay the dataset label and the AI output and split their union into connected regions: a lesion labelled but missed by the AI, an AI finding with no label, an extent disagreement, or an agreement (control). You are never told which.

## 6. Trial sample

- 15 photographs drawn at random within strata: DDR grade 0–1 (3), DDR grade 2–4 (5), IDRiD (3), TJDR standard-field (2), TJDR ultra-widefield (2).
- In each photograph about three regions drawn at random (one AI-only, one label-only, one agreement) plus about ten regions chosen to show typical errors: about 55 regions in all.
- Regions in the same photograph never overlap, so revealing one region's outlines never shows the answer for another.
- The design and random seed are fixed before reading.

## 7. What a reader does

Everything happens in one offline web page that opens from the data folder by double-click (no installation, no internet). For each photograph:

| Step | What you see | Questions |
|---|---|---|
| Grade the image | The photograph only | Image quality; ICDR grade; DME (from HEX relative to the fovea); optional: IRMA, NVD/NVE, venous beading, preretinal or vitreous hemorrhage, laser scars; click the fovea (or "not visible"); optional: click the disc |
| Region, step A | A dashed circle, no outlines | What is in the circle; confidence |
| Reveal | Dataset label (solid) and AI (dashed) inside the region's window; optional heat map | Your step-A answer is locked |
| Region, step B | Both outlines | Who is right; what is wrong with the label or the AI; what the error would change under ICDR; would it change what you do for this patient; optional comment |
| Whole image | All outlines | Grade and referral from the AI lesions alone; lesions missing from the dataset label; quality of the label and of the AI map; agreement with the dataset's own grade where available |

A short feedback page at the end asks which questions were useful and which were not, which options were missing or unclear, and whether the two consequence questions capture clinical consequence.

## 8. Clinical consequence, defined by ICDR

**What would change under ICDR** (several may apply; "Nothing" excludes the others), judged against your own grading of the image:

| Option | When it applies |
|---|---|
| Nothing | Grade and DME unchanged (for example, many lesions of the same type elsewhere) |
| No DR ↔ mild NPDR | The only MA in the eye |
| Mild ↔ moderate NPDR | The only HG, HEX or CWS in an MA-only eye; this also crosses the referral threshold |
| Moderate ↔ severe NPDR | The hemorrhage criterion of the 4-2-1 rule (more than 20 HG in each of four quadrants). Of the four lesion types, only HG can reach severe NPDR |
| DME status | HEX in the macula or at the foveal center changes DME (none / non-center-involving / center-involving) |

**Would it change what you do for this patient?** No (for example, a small lesion far from the fovea) / follow-up interval / referral or treatment / unsure.

Comparing the two shows where a rule-based threshold is crossed but clinical practice would not change.

## 9. Analysis

- **Who is right (primary):** design-weighted shares of AI-only regions that you identify as the lesion the AI reported (label omissions), and of labelled regions you confirm (true AI misses), by lesion type, with 95% intervals from a bootstrap over images. Primary results use the blinded step-A answers.
- **Clinical consequence:** frequencies of each ICDR consequence and of management change, and their cross-tabulation.
- **Distance from the fovea:** each region's distance from your foveal click in disc diameters (1 DD = fovea-to-disc distance / 2.5 when you click the disc), in three bands fixed before reading: ≤1 DD, 1–3 DD, >3 DD. We report label omissions, ICDR consequence and management change by band.
- **Secondary:** what went wrong (counts); probability that the label is wrong given how many models disagree with it; lesion-level precision and recall after correcting label errors; agreement between readers (formal batch); grade and referral implied by the AI lesions versus yours.
- The trial has one reader and is used only to revise the questions and the tool; its answers are not study results.

## 10. Time commitment

| Activity | Time |
|---|---|
| Trial: 15 images, about 55 regions | about 1 hour (can be split) |
| Trial feedback discussion | 30 min |
| Formal batch, per reader (size fixed after the trial) | about 4.5–7 h for about 100 images |

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
| 1–4 Nov | Trial folder sent by e-mail |
| 4–10 Nov | Trial reading and feedback |
| 11–14 Nov | Questions revised; formal batch and readers fixed |
| From 15 Nov | Formal batch |
