# BTC DVOL Competition

This repository contains participant-facing documentation and examples for the BTC DVOL (Bitcoin 30-day implied volatility) competition.

- Package repo: https://github.com/jberros/btcvol-python
- Install package: `pip install btcvol`
- Documentation site content: [docs/index.md](docs/index.md)
- Starter notebook: [notebooks/Getting_Started.ipynb](notebooks/Getting_Started.ipynb)

## Quick Start

1. Install the package:

```bash
pip install btcvol
```

2. Open the starter notebook:

```bash
jupyter notebook notebooks/Getting_Started.ipynb
```

3. Implement your tracker by subclassing `TrackerBase` and run a local test.

## What You Are Predicting

This competition is about forecasting the 30-day implied volatility of Bitcoin (Deribit DVOL). This is not a price prediction task.

For full details, see:
- [docs/competition_overview.md](docs/competition_overview.md)
- [docs/scoring.md](docs/scoring.md)
- [docs/submission.md](docs/submission.md)

## Repo Structure

- `docs/` - Participant documentation (overview, getting started, scoring, submission)
- `notebooks/` - Starter notebooks
- `README.md` - This file
