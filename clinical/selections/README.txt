Frozen selections (selection.json = the key) returned from the server; committed as the record of the draw (docs/clinical/28 section 5).
A formal batch's selection is the answer key: it is committed here only when its review closes; until then its sha256 is recorded in docs/clinical/35 (batch01) and the file stays in runs/CLINREV/selection/ (not in git). The frozen design already fixes the draw.
