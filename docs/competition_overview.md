# BTC Implied Volatility Prediction Competition

This competition challenges participants to predict the 30-day implied volatility (DVOL) of Bitcoin using the Deribit volatility index.

## About

The BTC DVOL Competition tests your ability to forecast Bitcoin's 30-day implied volatility, a forward-looking measure derived from options prices that reflects market expectations of future price movement. Unlike price prediction, this competition focuses on volatility forecasting, a critical component of options pricing and risk management.

This is not a price prediction competition. The target is Deribit's DVOL index, which represents the market's consensus view of Bitcoin's volatility over the next 30 days.

The competition uses real-time data from Deribit's public API to evaluate predictions against actual implied volatility measurements.

## Quick Start

Install the package and run the starter notebook:

```bash
pip install btcvol
jupyter notebook ../notebooks/Getting_Started.ipynb
```

## Dataset

### Data Sources

1. Deribit Volatility Index API
   - Target DVOL (30-day implied volatility) for scoring
   - Endpoint: `/public/get_volatility_index_data`

2. CrunchDAO Price API (optional for model development)
   - Historical and real-time Bitcoin price data
   - You can also use other public sources (for example Pyth Network)

### Data Structure

Each prediction requires:
- asset: "BTC" (Bitcoin only)
- horizon: 3600s (1 hour) or 86400s (24 hours)
- step: 900s (15 minutes)
- predicted_value: your forecasted implied volatility (0-1 scale)

### Target Variable

The target is the Deribit DVOL (30-day implied volatility index). It is updated in real time as option prices change and expressed as a decimal (for example 0.40 = 40% annualized volatility).

## Tournament Structure

### Timeline

Season 1:
- Training: February 16 - March 13, 2026
- Scoring: March 16 - April 17, 2026
- Competition End: April 17, 2026

### Prediction Frequency

Models must generate predictions every 15 minutes for both horizons.

## Useful Links

- Package repo: https://github.com/jberros/btcvol-python
- Deribit DVOL API: https://docs.deribit.com/#public-get_volatility_index_data
- CrunchDAO forum: https://forum.crunchdao.com
