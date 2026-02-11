# Scoring

Participants are evaluated using mean squared error (MSE) between predicted DVOL and the actual DVOL.

```python
error = predicted_dvol - actual_dvol
mse = error ** 2
```

Lower MSE is better.

## Scoring Windows

The official competition score uses a 30-day window matching the DVOL target. Additional windows are provided for monitoring:

- 30D Final Score (official)
- 14D Steady Score
- 7D Live Score
- 24H Short-term Score
- 4H Short-term Score
- 1H Recent Score

## Prediction Frequency

Models must generate predictions every 15 minutes (step = 900 seconds) for both horizons:

- 1 hour horizon (3600s)
- 24 hour horizon (86400s)
