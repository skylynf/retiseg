# Reader guide

RetiSeg clinical review, formal batch. Version 2.1, 6 November 2026. About ten minutes to read; keep it open while you start.

New since the trial: an ETDRS grid on the photograph, with a question on where the closest hard exudate lies; the dataset label and the AI are judged in two separate questions; IRMA and new vessels can be marked; a question on your management for each eye; for each image you first answer step A for every region, then reveal all outlines at once. The question on the AI's grade and referral is gone.

The batch has 100 photographs in two parts: part 1 (20 photographs, projected 1.5–2 hours) now, part 2 (80) after part 1 is back. Each part is its own folder and saves its answers separately.

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
- **Marker** (the dashed circle, **K**): on or off at any time.
- **Outline controls** (after the reveal of an image): **Dataset label** (solid lines, **D**), **AI** (dashed lines, **M**), **Heat map** (the AI's probability, **H**), and one switch per lesion type. Flipping the dataset label and the AI on and off is the quickest way to compare them.
- **ETDRS grid** (**E**): appears once you have marked the fovea (section 6); shown by default when you grade the image and in the whole-image step, hidden in the region steps.
- **Right:** the questions, one numbered card each. A card's number turns dark when answered; a card turns amber if a required answer is missing. **Panel: light / dark** in the top bar switches the colors of this area. Keys **1–9** choose an option of the first unanswered question; **Enter** moves on.

Lesion colors: **MA cyan**, **HG magenta**, **HEX yellow**, **CWS green**. Dotted gray lines, where present, mark areas the dataset itself labeled as uncertain.

## 4. The steps for each photograph

Each photograph has five parts, always in this order: grade the image; step A for every marked region; one reveal; step B for every region; the whole image.

### Grade the image (no outlines, no regions)

- **Image quality.**
- **Foveal center** and **optic disc center** (both required): click *Set*, then click the point on the image; choose *Not visible* if you cannot see it. The ETDRS grid appears, centered on your foveal mark, as soon as you mark the fovea; your disc mark sets its scale.
- **ICDR grade.** No DR; mild NPDR (MA only); moderate NPDR (more than MA only, less than severe); severe NPDR (4-2-1 rule: more than 20 intraretinal HG in each of four quadrants, or definite venous beading in two or more quadrants, or prominent IRMA in one or more quadrants; no PDR); PDR (NV, and/or vitreous or preretinal hemorrhage); cannot grade.
- **Closest hard exudate to the fovea** (a location on the grid): none on the grid; outer ring only (1.5–3 mm from the fovea); inner ring (0.5–1.5 mm, within 1 DD); central subfield (within 0.5 mm); cannot assess (also when the fovea is not visible). This records where the exudate lies; it is not a DME diagnosis (section 6).
- **Signs beyond the four lesion types** (optional): IRMA, NVD/NVE, venous beading, preretinal or vitreous hemorrhage, laser scars, other. If you tick IRMA or NVD/NVE, you can mark them: click *Add IRMA* (or *Add NV*), click each location on the image, then *Done* or Esc. *Undo last* removes the last mark.
- **Your management for this eye, from this photograph**: routine rescreening; shorter follow-up or monitoring; refer (non-urgent); refer urgently; cannot decide. Answer as you would act in practice.

These answers lock when you move on to step A, because the circles show where the label or the AI drew lesions. Check your fovea and disc marks before moving on.

### Step A, every region in turn (blinded)

A dashed white circle marks one region at a time. Somewhere inside it the dataset label, the AI or both drew one of the four lesion types; you are not told which, or which type. The circle's size, and the zoom, follow the size of what was drawn. Many regions in this batch lie near the fovea or look like vessels; that is by design, so please do not let it influence your answer. Look inside the circle (zoom and red-free help) and answer from the photograph alone.

- **Main finding**: one of the four lesion types (MA, HG, HEX, CWS), another retinal finding (including IRMA and NV), normal anatomy, an artifact, or *Cannot determine*.
- **Confidence**: certain, probable, possible.

*Next region* takes you to the next circle. No outline of the image is shown until you have answered step A for all its regions; until then you can go back and change them.

### Reveal

On the last region, the button reads **Reveal outlines**. Clicking it locks your step A answers for all regions of this image. These blinded answers are our main measurement.

### Step B, every region in turn (judge both outlines)

The dataset label (solid) and the AI (dashed) appear inside the square around the region. The square may also show outlines of a neighboring region; your answers for that one are already locked. The top of the panel repeats your step A answer and gives the ETDRS location of the circle's center (for example "outer ring, nasal (N2), 2.6 mm from the fovea"); part of the lesion may lie closer to the fovea.

- **Dataset label in this region** and **AI in this region**, each: *Correct* (including correctly drawing nothing); *Right lesion, wrong extent* (too large, too small, or several lesions merged); *Wrong lesion type*; *Marks a non-lesion*; *Misses a lesion*; *Can't decide* (then: depends on definition, or image quality or resolution). If both are wrong, say so in both questions.
- If either outline is not correct: **which threshold the error would cross** (next section) and **if the wrong outline were used, would your management of this patient differ?** (no; follow-up interval; referral or treatment; unsure). The second is your own clinical judgement; compare it with your management for the eye in the grading step.
- **Comment** (optional). Short notes are very valuable when no option fits.

### Whole image (all outlines, grid shown)

- **Lesions missing from the dataset label** for the whole image: none; a few, outside the grid; a few, including on the grid; many.
- **Quality** of the dataset label and of the AI lesion map for this image.
- For DDR images the dataset's own ICDR grade is shown; say whether you agree.

## 5. Which threshold the error would cross

Judge against your own grading of this image, as if the wrong outline were used. Several options may apply; *None* excludes the others. If both outlines are wrong, tick what either error would change.

| Option | Applies when | Example |
|---|---|---|
| None | ICDR grade and hard-exudate band unchanged | One of thirty HG in a moderate NPDR eye |
| No DR ↔ mild NPDR | The only MA in the eye | AI marks a single MA the label missed in an otherwise normal eye |
| No DR ↔ moderate NPDR | The only HG, HEX or CWS in an eye without MA | AI marks a single HG the label missed in an eye with no other lesion |
| Mild ↔ moderate NPDR | The only HG, HEX or CWS in an MA-only eye | Label missed the only small HG in an eye with a few MA |
| Moderate ↔ severe NPDR | Any 4-2-1 criterion | Missed HG that decide whether the fourth quadrant passes 20; IRMA outlined as HG |
| NPDR ↔ PDR | New vessels or preretinal hemorrhage outlined as HG | NVE outlined as a hemorrhage by the AI |
| Closest HEX moves into or out of 1 DD | The inner-ring boundary, 1.5 mm from the fovea | AI misses the only HEX at 1 mm from the fovea |
| Closest HEX moves into or out of the central subfield | The 0.5 mm boundary | Label misses HEX at the foveal center |

These options describe rules and locations. Whether the error would change what you do is the next question, and only you can answer it: a rule can be crossed without any change in management, and the reverse; that difference is exactly what we want to see.

## 6. The ETDRS grid and hard exudates

The grid is centered on your foveal mark. Its scale takes your fovea-to-disc distance as 4.7 mm (the adult average is about 4.6–4.8 mm); until you mark the disc, or if you mark it as not visible, the folder's own estimate of the scale is used. Circles: central subfield C (1 mm diameter), inner ring (1–3 mm), outer ring (3–6 mm); the 45° lines divide the rings into superior (S), inferior (I), nasal (N, the disc side) and temporal (T) subfields, labeled S1–T1 and S2–T2. The fovea-to-disc distance varies with an SD of about 7%, so in some eyes the circles may be 15–20% off; treat them as a guide. 1 DD = 1.5 mm is the ETDRS convention; real discs average about 1.8 mm, so "within 1 DD" judged against the disc itself reaches a little beyond the inner ring. If the fovea is not visible there is no grid; answer the hard-exudate question with *Cannot assess*.

The hard-exudate question records a location only. A single non-stereo color photograph cannot show retinal thickening (stereo photographs or OCT can), so it cannot establish diabetic macular edema. For reference: UK diabetic eye screening grades exudate within 1 DD of the center of the fovea as referable maculopathy (M1); the ETDRS counts hard exudate within 500 µm of the center as clinically significant macular edema only when the adjacent retina is thickened; in DRCR Retina Network trials, center-involved DME on OCT is central subfield thickness ≥320 µm (men) or ≥305 µm (women) on Spectralis.

## 7. Definitions we use

- **MA**: small, round, red dot with sharp margins, typically smaller than the width of a major vein at the disc margin (about 125 µm). Larger or irregular red lesions are HG. Where you cannot tell MA from a small dot HG: in step A choose the more likely one with confidence *Possible*; in step B answer *Can't decide*, then *Depends on definition*.
- **HG**: intraretinal dot, blot or flame hemorrhage. Preretinal or vitreous hemorrhage is a separate option.
- **IRMA** and **NV**: when the AI or the label outlines a tortuous or abnormal vessel as HG, answer what you see (IRMA, NV, or a normal vessel) in step A; in step B the outline is then *Wrong lesion type* or *Marks a non-lesion*.
- **HEX**: yellow-white, sharply defined deposits, often in clusters or rings. Distinguish from drusen and reflexes.
- **CWS**: white, fluffy, ill-defined lesion of the nerve fiber layer. Distinguish from nerve-fiber-layer sheen and myelinated nerve fibers.
- **Light reflex**: reflections from the internal limiting membrane, the foveal reflex or vessel walls.
- **Artifact**: dust or smudges on the lens, eyelashes, flare, the edge of the field, distortion at the periphery of ultra-widefield images.

## 8. Good practice

- We expect about 4–6 minutes per image (a projection; part 1 will check it). Sittings of 10–15 images work well; take breaks whenever you like. Your image order is random, so stopping early still leaves a useful sample.
- Please work only through `index.html`. The data files in the folder also contain the outlines, which the page shows only after your blinded answers.
- Please do not discuss your answers with other readers. We will not discuss your part 1 answers with you until part 2 is finished, apart from problems with the page or the questions.
- Please do not search for the images elsewhere.
- If the page misbehaves, export your results first, then tell us what happened (a screenshot helps).
- Please do not share the folder, and delete it when we tell you the review is closed.

## 9. Keys

| Key | Action |
|---|---|
| 1–9 | Choose an option of the first unanswered question |
| Enter | Next; on the last step A of an image, reveal outlines |
| F, Z | Fit the image; zoom to the region |
| +, −, wheel, drag | Zoom and move |
| G | Red-free view |
| E | ETDRS grid on or off (after marking the fovea and disc) |
| D, M, H, K | Dataset label, AI, heat map, marker (when available) |
| ? | Help, including an ICDR quick reference |
| Esc | Cancel placing a point; stop adding IRMA / NV marks; close a dialog |
