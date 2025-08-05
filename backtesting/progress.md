# Current status

## Models

- Trained a classification model for forward 300s prediction, based on lookback 120s data
- With the same features, trained another three regime based classification models. Also a clustering model for identifying regimes
- Trained a clustering model for identifying traders

## Pipelines

- Reusable feature engineering pipeline for classification model related features
- Reusable inference pipeline for classification model and regime detection model

## Backtesting with Backtrader lib

- Basic implementation of data feed, broker and analyzer for per transaction based data
- Implementation of strategy for basic classification model
- Use Backtrader lib as a wrapper for running the test and generating the results for analysis
- Implement actual strategies into data feed generation pipeline

## Backtesting result

- First round
```
Average return: 102.78%
Median return: 8.95%
Best performer: GUD8vxk5LSNGvB4YKUnekz8oqGurgXKF6iFQpeNrWPGG (6197.95%)
Worst performer: 7tPPYTBKrFLKKnoCwijrsfjAYadyp7GpAmSPUbVwbonk (-13.74%)
Coins with trades: 806/822
```

# Need improvement

## Models

- AUC scores of all classification models are not ideal, between 0.66 to 0.7, and not good enough for identifying positive class (buy signal)
- Should try adding some features related with volitility and price actions from look back window

## Pipelines

- Feature engineering for trader clustering is built in a notebook, need refactor to modular codes
- Implement different strategies with feature engineering pipeline (for backtesting only)

## Backtesting with Backtrader lib

- ~~No log data persistance from backtesting run~~
- ~~Visualization and logging still have some issues~~
- ~~The implementation for per transaction based data seems problematic because this lib is built for OHLVC format data~~
- Should just use this lib for basic functions: ingest data feed, execute orders strictly based on signals (from data feed), log run results and generate visualization 

# TODO

- ~~A few more model error analysis (model eval on segmentation of validation data) to identify current classification model's strength and weakness, so we could design a best strategy based on current model~~ ---- DONE
- Backtest the strategy with the current backtesting framework
    - ~~Using the validation data~~ ---- DONE
    - Pulling more new data (new coins' first day trades) from Flipside, and use them to test
- ~~If the backtrader framework turns out to be a headache than a solution, consider two options:~~
    1. ~~Convert the per transaction data into a 30s based OHLVC data, and use most of the default setups from backtrader to run backtesting~~
    2. ~~Rebuild a custom pipeline for backtesting~~
- Come up with more features for classification model, and use different forward windows for target variables
- 