# Results

This directory is filled by running on a GPU:

```bash
make bench       # results/bench.json, bench.md, roofline.png
make evaluate    # results/evaluation.md (needs agent runs in runs/)
```

or everything at once with `./scripts/run_all.sh`.

No numbers are committed yet: the code was written and type-checked on a
machine without a GPU, and results are only reported after they are measured.
