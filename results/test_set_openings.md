# `test` split openings

BLOCK 3.6 / 17.4: the `test` split is opened **once**, at the end, and every
opening is logged here with the date and reason.
`scripts/eval_detection.py --split test` appends a row automatically and refuses
a second opening without `--allow-reopen`.

Thresholds and every policy parameter are fitted on `val` only.

| date (UTC) | model | policy | reason |
|---|---|---|---|

_No openings yet - the evaluation set is not labelled, so `test` has never been
run._
