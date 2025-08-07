# Initial thoughts

- Not trying to build any physical prediction models, because it would be hard to scale
- Will only use PA and volume data to build models
- Ideally, model should be able to find mispricing from three types of signals:
    - Insider buying
    - Impulsive move and mean reversion
    - Decay speed
- Should use LLM to classify the events into different types:
    - With insider potential
    - Knowledge based bet
    - Pure luck
    - News driven 