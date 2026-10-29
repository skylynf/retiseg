# Reader guide

RetiSeg clinical review, trial. Version 1.0, 29 October 2026. About ten minutes to read; keep it open while you start.

## 1. Opening the folder

1. Unzip the folder you received and keep its contents together (`index.html`, `pack.js`, `images/`, `heat/`).
2. Double-click `index.html`. Chrome, Edge or Firefox work best; use a screen of 13 inches or more.
3. If the page says it cannot find the pack, click **Open pack folder** and choose the unzipped folder.
4. Enter your **reader code** (given to you by the study team) and click **Start**. Use the same code every time; that is how the page finds your saved answers.

## 2. Saving and returning your answers

- Every click is saved automatically in this browser on this computer. You can close the page and continue later with the same reader code.
- At the end of **every sitting**, click **Export results** (top right) and e-mail us the downloaded `.json` file. The final page, *Finish and export*, does the same.
- In Chrome or Edge you can also choose **Save & import → Save continuously to a file**; the page then keeps that file up to date.
- To move to another computer, open the folder there, enter your code, choose **Save & import**, and import your last exported file.
- Private or incognito windows forget everything when closed; please do not use them.

## 3. The screen

- **Left:** the list of photographs; a dot fills as you go.
- **Centre:** the photograph. Scroll to zoom, drag to move, **F** to fit the whole image, **Z** to return to the current region. The small map at the bottom right shows where you are; click it to jump.
- **Image controls:** brightness, contrast and **Red-free** (green channel, **G**). *Reset* undoes the adjustments.
- **Outline controls** (after a reveal): **Dataset label** (solid lines, **D**), **AI** (dashed lines, **M**), **Heat map** (the AI's probability, **H**), **Marker** (the dashed circle, **K**), and one switch per lesion type. Flipping the dataset label and the AI on and off is the quickest way to compare them.
- **Right:** the questions, one numbered card each. A card's number turns dark when answered; a card turns amber if a required answer is missing. **Panel: light / dark** in the top bar switches the colours of this area. Keys **1–9** choose an option of the first unanswered question; **Enter** moves on.

Lesion colours: **MA cyan**, **HG magenta**, **HEX yellow**, **CWS green**. Dotted grey lines, where present, mark areas the dataset itself labelled as uncertain.

## 4. The steps for each photograph

### Grade the image (no outlines)

- **Image quality.**
- **ICDR grade.** No DR; mild NPDR (MA only); moderate NPDR (more than MA only, less than severe); severe NPDR (4-2-1 rule: more than 20 intraretinal HG in each of four quadrants, or definite venous beading in two or more quadrants, or prominent IRMA in one or more quadrant; no PDR); PDR (NV, and/or vitreous or preretinal hemorrhage).
- **DME**, judged on the photograph from HEX relative to the foveal center: no DME, non-center-involving, center-involving, cannot assess.
- **Signs beyond the four lesion types** (optional): IRMA, NVD/NVE, venous beading, preretinal or vitreous hemorrhage, laser scars. These are not labelled in the datasets but drive severe NPDR and PDR.
- **Foveal center** (required): click *Set*, then click the fovea; choose *Not visible* if you cannot see it.
- **Optic disc center** (optional): gives the scale in disc diameters.

### Each marked region, step A (blinded)

A dashed white circle marks a region. It may be a lesion the AI missed, an AI finding with no label, a disagreement about extent, or a place where AI and label agree; you are not told which. Look inside the circle (zoom and red-free help) and answer from the photograph alone.

- **Main finding**: one of the four lesion types (MA, HG, HEX, CWS), another retinal finding, normal anatomy, an artifact, or *Cannot determine*.
- **Confidence**: certain, probable, possible.

When you click **Reveal outlines**, your step-A answer is locked. This blinded answer is our main measurement.

### Each marked region, step B (outlines shown)

The dataset label (solid) and the AI (dashed) appear inside this region's window only.

- **Who is right?** Both acceptable; AI right, dataset label wrong; dataset label right, AI wrong; both wrong; depends on definition (for example MA versus a small dot HG); cannot tell from this photograph.
- **What is wrong** with the dataset label or with the AI (tick all that apply). For an AI false alarm, what it really is was already captured in step A.
- **What would change under ICDR** (section 5).
- **Would it change what you do for this patient?** No; follow-up interval; referral or treatment; unsure.
- **Comment** (optional). Short notes are very valuable when no option fits.

### Whole image (all outlines)

- **Grade and referral from the AI lesions alone**, compared with yours. Referable means moderate NPDR or worse and/or DME.
- **Lesions missing from the dataset label** for the whole image: none; a few, mostly far from the fovea; a few, including near the fovea; many.
- **Quality** of the dataset label and of the AI lesion map for this image.
- Where the dataset provides its own grade, it is shown; say whether you agree.

## 5. What would change under ICDR

Judge against your own grading of this image, as if the wrong outline (AI or label, as you decided) were used. Several options may apply; *Nothing* excludes the others.

| Option | Applies when | Example |
|---|---|---|
| Nothing | Grade and DME unchanged | One of thirty HG in a moderate NPDR eye |
| No DR ↔ mild NPDR | The only MA in the eye | AI marks a single MA the label missed in an otherwise normal eye |
| Mild ↔ moderate NPDR | The only HG, HEX or CWS in an MA-only eye (crosses the referral threshold) | Label missed the only small HG in an eye with a few MA |
| Moderate ↔ severe NPDR | Hemorrhage criterion of the 4-2-1 rule (>20 HG in each of four quadrants) | Missed HG that decide whether the fourth quadrant passes 20 |
| DME status | HEX in the macula or at the foveal center | AI misses HEX at the center (center-involving becomes non-center-involving) |

The second question records what you would actually do. A lesion can cross a threshold under the rule and still not change management, for example a small isolated HG far from the fovea; that difference is exactly what we want to see.

## 6. Definitions we use

- **MA**: small, round, red dot with sharp margins, typically smaller than the width of a major vein at the disc margin (about 125 µm). Larger or irregular red lesions are HG; where you cannot tell, answer *Depends on definition*.
- **HG**: intraretinal dot, blot or flame hemorrhage. Preretinal or vitreous hemorrhage is a separate option.
- **HEX**: yellow-white, sharply defined deposits, often in clusters or rings. Distinguish from drusen and reflexes.
- **CWS**: white, fluffy, ill-defined lesion of the nerve fibre layer. Distinguish from nerve-fibre-layer sheen and myelinated nerve fibres.
- **Light reflex**: reflections from the internal limiting membrane, the foveal reflex or vessel walls.
- **Artifact**: dust or smudges on the lens, eyelashes, flare, the edge of the field, distortion at the periphery of ultra-widefield images.

## 7. Good practice

- About one minute per region is typical; take breaks whenever you like.
- Please do not search for the images elsewhere.
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
| D, M, H, K | Dataset label, AI, heat map, marker (when available) |
| ? | Help, including an ICDR quick reference |
| Esc | Cancel placing a point; close a dialog |
