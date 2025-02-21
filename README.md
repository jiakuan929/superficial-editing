# superficial-editing
This is the official implementation for superficial editing.

## Evaluation
`python ./evaluation/run.py --model $model --editor $editor`

## Analysis
Intervention on the residual stream:

`python ./mechanistic_analysis/patch_resid.py --model $model --editor $editor --patch $patch_position`

## Related Repositories
[EasyEdit](https://github.com/zjunlp/EasyEdit/tree/main)
