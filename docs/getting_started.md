# Getting Started

This guide helps you set up locally and validate your model before submitting.

## Install

```bash
pip install btcvol
```

## Create a Tracker

Implement a tracker by subclassing `TrackerBase` and returning a list of volatility predictions for the requested horizon and step.

```python
from btcvol import TrackerBase, test_model_locally
import numpy as np

class MyVolatilityTracker(TrackerBase):
    def predict(self, asset: str, horizon: int, step: int) -> list:
        n_steps = max(1, horizon // step)
        base_vol = 0.42
        return [float(base_vol)] * n_steps

# Run local tests
if test_model_locally(MyVolatilityTracker):
    print("Model is ready for submission")
```

## Notebook

Open the starter notebook for a step-by-step walkthrough:

```bash
jupyter notebook ../notebooks/Getting_Started.ipynb
```
