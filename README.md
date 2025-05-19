# superficial-editing
This is the official implementation for superficial editing.

## Evaluation
`python ./evaluation/run.py --model $model --editor $editor`

## Analysis
Experiments for Figure 2:

`python ./mechanistic_analysis/patch_resid.py --model $model --editor $editor --patch $patch_position`

Experiments for Figure 3:

`python ./mechanistic_analysis/module_io_prob.py --model $model --editor $editor`

Experiments for Hypothesis 1:

```
python ./mechanistic_analysis/suppress2.py --model $model --editor $editor
python ./mechanistic_analysis/prove_noold.py --model $model --editor $editor
```

## Related Repositories
[EasyEdit](https://github.com/zjunlp/EasyEdit/tree/main)
