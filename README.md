# Superficial Editing
This is the code implementation for our paper "Revealing the Deceptiveness of Knowledge Editing: A Mechanistic Analysis of Superficial Editing".

## Evaluation
`python ./evaluation/run.py --model $model --editor $editor --dataset $dataset --datatype $datatype`

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

Experiments for Figure 6:

```
python ./mechanistic_analysis/patch_wo_attn.py --model $model --editor $editor --patch $patch_position
```

Experiments for Figure 7:

```
python ./mechanistic_analysis/layer_head_visualization.py --model $model --editor $editor
```

Experiments for Ablating identified attention heads:

```
python ./mechanistic_analysis/ablate_head.py --model $model --editor $editor
```

Experiments for DSR:

```
python ./mechanistic_analysis/extract_rate.py --model $model --editor $editor --locate_top $topp --decode_top $topk
```

Experiments for ablating identified vectors:

```
python ./mechanistic_analysis/ablate_svd.py --model $model --editor $editor --locate_top $topp
```
