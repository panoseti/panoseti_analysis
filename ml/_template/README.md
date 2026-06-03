# ml/\_template — batteries-included model starter

Copy this directory to `ml/<your-model>/` to start a new model:

```bash
cp -r ml/_template ml/my-new-model
```

## What's included

| File                      | Purpose                                               |
| ------------------------- | ----------------------------------------------------- |
| `notebooks/lab.py`        | Volatile R&D layer — edit freely with `%autoreload 2` |
| `recipes/my_model_v1.yml` | Science params (hyperparams + sweep space)            |

## The promotion gradient

```
ml/_template/notebooks/lab.py  (volatile, copy + hack)
    ↓ once the model architecture stabilises
src/panoseti_analysis/algorithms/<yourfile>.py  +  @register_model("id")
    + a recipe in recipes/<name>.yml
    ↓ once the training loop is general enough
src/panoseti_analysis/adapters/ml/bench.py  (shared utilities)
```

## Batteries included (no re-implementation needed)

```python
from panoseti_analysis.adapters.ml.bench import (
    get_device,   # auto-select device
    quick_train,  # load feature cache → train → best model
    plot_history, # training-curve plot
    plot_eval,    # confusion matrix + PR curve
)
```

## Quick start

1. **Register your model** in `lab.py` (or in `algorithms/`):

   ```python
   @register_model("my_model_v1")
   class MyModel(nn.Module): ...
   ```

2. **Train** (notebook cell or script):

   ```python
   model, result = quick_train(FEATURE_CACHE, {"arch": "my_model_v1", "epochs": 10})
   plot_history(result)
   plot_eval(model, x_val, y_val)
   ```

3. **Full Ray Train run** once the recipe is ready:

   ```bash
   pa-train-cloud FEATURE_CACHE.zarr models/ --recipe recipes/my_model_v1.yml
   ```

4. **Hyperparameter sweep**:
   ```bash
   pa-tune-cloud FEATURE_CACHE.zarr models/ --recipe recipes/my_model_v1.yml
   ```

## Tracking (optional)

Set `WANDB_API_KEY` in your environment and add `wandb_project: <project>` to the recipe.
