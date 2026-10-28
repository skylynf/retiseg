# Reader guide

RetiSeg clinical review, pilot. Version 0.9, 28 October 2026. About ten minutes to read; keep it open while you start.

## 1. Opening the folder

1. Unzip the folder you received and keep its contents together (`index.html`, `pack.js`, `images/`, `heat/`).
2. Double-click `index.html`. Chrome, Edge or Firefox work best; use a screen of 13 inches or more.
3. If the page says it cannot find the pack, click **Open pack folder** and choose the unzipped folder.
4. Enter your **reader code** (R1, R2, …, given to you by the study team) and click **Start**. Use the same code every time; that is how the page finds your saved answers.

## 2. Saving and returning your answers

- Every click is saved automatically in this browser on this computer. You can close the page and continue later with the same reader code.
- At the end of **every sitting**, click **Export results** (top right) and send us the downloaded `.json` file. The final page, *Finish and export*, does the same.
- In Chrome or Edge you can also choose **Save & import → Save continuously to a file** once; the page then keeps that file up to date.
- To move to another computer, open the folder there, enter your code, choose **Save & import**, and import your last exported file.
- Private or incognito windows forget everything when closed; please do not use them.

## 3. The screen

- **Left:** the list of photographs, with a dot that fills as you go.
- **Centre:** the photograph. Scroll to zoom, drag to move, **F** to fit the whole image, **Z** to return to the current region. The small map at the bottom right shows where you are; click it to jump.
- **Image controls:** brightness, contrast and **Red-free** (green channel, **G**), which helps with microaneurysms and haemorrhages. *Reset* undoes the adjustments.
- **Outline controls** (only after a reveal): **Dataset** (solid lines, **D**), **Model** (dashed lines, **M**), **Heat map** (the model's probability, **H**), **Marker** (the dashed circle, **K**), and one switch per lesion type. Flipping the dataset and model outlines on and off is the quickest way to compare them.
- **Right:** the questions for the current step. Keys **1–9** choose an option of the first unanswered question (the numbers are shown on the buttons); **Enter** moves on.

Lesion colours: **MA cyan**, **HE magenta**, **EX yellow**, **SE green**. Dotted grey lines, where present, mark areas the dataset itself labelled as uncertain; they do not count in the scoring.

## 4. The steps for each photograph

### Image assessment (no outlines)

Grade the photograph as you would in clinic.

- **Image quality** and, if not good, what limits it.
- **ICDR DR grade.** 0 no apparent retinopathy; 1 mild NPDR (microaneurysms only); 2 moderate NPDR (more than microaneurysms only, less than severe); 3 severe NPDR (any of: more than 20 intraretinal haemorrhages in each of four quadrants, definite venous beading in two or more quadrants, prominent IRMA in one or more quadrants; no signs of PDR); 4 PDR (neovascularisation and/or vitreous or pre-retinal haemorrhage).
- **DME (photographic surrogate of the ICDR DME scale).** Use hard exudates and any other photographic sign of thickening, relative to the foveal centre: *absent*; *mild* (in the posterior pole, distant from the centre); *moderate* (approaching the centre, not involving it); *severe* (involving the centre).
- **Other findings** (optional).
- **Foveal centre** and **optic disc centre**: click *Set*, then click on the image. Skip if not visible.

### Each marked region, step A (blinded)

A dashed white circle marks a region. You are not told why it was chosen: it may be a lesion the model missed, a finding only the model made, a disagreement about extent, or a place where model and dataset agree. Look inside the circle (zoom and red-free help) and answer from the photograph alone.

- **Main finding in the circle.** If several things are present, choose the most important and tick *more than one distinct lesion*. Choose *Cannot determine* when the photograph does not allow a call.
- **Confidence**: certain, probable, possible.

When you click **Reveal outlines**, your step-A answer is locked. This is deliberate: it is our unbiased measurement.

### Each marked region, step B (outlines shown)

The dataset annotation (solid) and the model (dashed) appear inside this region's window only.

- **Main explanation of the difference.** Pick one:
  - *No real difference*: both outlines acceptable (expected for many control regions).
  - *Dataset annotation wrong, model right.*
  - *Model wrong, dataset annotation right.*
  - *Both wrong.*
  - *Definition-dependent*: reasonable graders could disagree (for example MA versus a small dot haemorrhage, or where one hazy patch ends).
  - *Cannot be resolved from this photograph*: quality or resolution is not enough.
- **What is wrong with the annotation / the model**: tick all that apply.
- **Difficulty for a human grader.**
- **Clinical consequence of this error, in this image** (see section 5).
- **Decisive on its own?** (see section 5).
- **What would most reduce this kind of error?** (optional).
- **Free text: what went wrong here?** Short notes are very valuable, especially when no option fits.

### Image summary (all outlines)

All outlines of the whole photograph are shown. Toggle them to compare.

- **Grading from the model's lesions alone**: if a grader saw only the model's lesion map, would the ICDR grade be the same as yours, lower or higher?
- **Referral from the model's lesions alone**: referable means moderate NPDR or worse and/or DME.
- **Overall quality** of the dataset annotation and of the model's map for this image.
- Where the dataset provides its own grade, it is shown; say whether you agree.

## 5. Clinical impact and "decisive": anchors

Judge the error **in this region**, assuming the wrong outline (model or dataset, as you decided) were used for grading this image, and everything else stays as you graded it.

| Level | Meaning | Examples |
|---|---|---|
| 0 None | No effect on any clinical decision | A vessel segment marked as haemorrhage in an image that is already PDR; an exudate outline 20% too large |
| 1 Minor | Changes lesion count or extent only | One of forty microaneurysms missed in a moderate NPDR image |
| 2 Moderate | Changes the DR grade or DME category, not the referral | Missed haemorrhages make a moderate NPDR image look mild, but DME already requires referral |
| 3 Major | Changes the referral decision or its urgency | The only haemorrhages in an otherwise mild image missed (moderate NPDR becomes mild, no referral); exudates at the fovea missed |

**Decisive** asks a different question: is this lesion the pivotal one, so that this single finding by itself decides a grading threshold? Examples: the only microaneurysm in the image (grade 0 versus 1); an exudate at the foveal centre (centre-involving DME); a frond of new vessels (PDR). An error can be decisive and still have low impact (for example the only microaneurysm, where grade 0 versus 1 does not change referral).

## 6. Definitions we use

- **Microaneurysm**: small, round, red dot with sharp margins, typically smaller than the width of a major vein at the disc margin (about 125 µm). Larger or irregular red lesions count as haemorrhages; where you cannot tell, say so (*Definition-dependent*).
- **Intraretinal haemorrhage**: dot, blot or flame-shaped red lesion. Pre-retinal or vitreous haemorrhage is a separate option.
- **Hard exudate**: yellow-white, sharply defined, often in clusters or rings. Distinguish from drusen (deeper, rounder, often near the fovea in older eyes) and from reflexes.
- **Cotton-wool spot**: white, fluffy, ill-defined lesion of the nerve fibre layer. Distinguish from nerve-fibre-layer sheen in young eyes and from myelinated nerve fibres.
- **Light reflex**: specular reflections from the internal limiting membrane, foveal reflex or vessel walls.
- **Artifact**: dust or smudges on the lens, eyelashes, flare, the edge of the field, distortion at the periphery of ultra-widefield images.

## 7. Good practice

- Work at your own pace; about one minute per region is typical. Take a break whenever you like.
- Do not discuss cases with other readers until the debrief meeting.
- Please do not search for the images elsewhere or look at other material about them.
- If the page misbehaves, export your results first, then tell us what happened (a screenshot helps).
- Please do not share the folder, and delete it when we tell you the review is closed.

## 8. Keys

| Key | Action |
|---|---|
| 1–9 | Choose an option of the first unanswered question |
| Enter | Reveal outlines / next |
| F, Z | Fit the image; zoom to the region |
| +, −, wheel, drag | Zoom and move |
| G | Red-free view |
| D, M, H, K | Dataset outline, model outline, heat map, marker (when available) |
| ? | Help |
| Esc | Cancel placing a point; close a dialog |
