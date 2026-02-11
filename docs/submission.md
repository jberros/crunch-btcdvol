# Submission

Submit your tracker to the CrunchDAO platform following the official competition submission workflow.

## Requirements

- A `TrackerBase` implementation
- Predictions for both 1h and 24h horizons
- Return exactly `horizon // step` values
- Run time within the competition limits

## Local Validation

Use the package test utilities before submitting:

```python
from btcvol import test_model_locally
from my_tracker import MyVolatilityTracker

test_model_locally(MyVolatilityTracker)
```

## Notes

- Failed predictions receive penalty scores
- Submissions can be updated during the training phase
- Once live scoring begins, model updates may be restricted
