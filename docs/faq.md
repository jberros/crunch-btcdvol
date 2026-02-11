# FAQ

## Is this a price prediction competition?

No. The target is the 30-day implied volatility index (Deribit DVOL).

## What assets are included?

Bitcoin only (BTC).

## What is the prediction format?

Return a list of floats in the 0-1 volatility range. The list length must equal `horizon // step`.

## What if my model fails during a round?

Failed predictions receive a penalty score.

## Where can I get help?

Use the CrunchDAO forum: https://forum.crunchdao.com
