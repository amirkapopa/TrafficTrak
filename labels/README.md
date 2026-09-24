# labels/

`dev_labels.json` holds **human-reviewed** development labels for sample
videos (`{video: [[start, end, label], ...]}`), created with
`scripts/annotate_candidates.py labels`.  Sample videos ship without labels;
nothing in this folder is ever generated automatically from model output
without a person accepting it.  No labels exist yet.
