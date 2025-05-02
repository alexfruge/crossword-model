# crossword-model

This project aims to develop a comprehensive test suite for evaluating the capabilities of large language models (LLMs) in solving crossword puzzle clues. The goal is to create a framework that can assess various aspects of LLM performance, including clue interpretation, word association, and logical reasoning.

## Features
- Feed in crossword clues to a suite of LLMs.
- Evaluate LLMs on solving crossword clues.
- Metrics for assessing accuracy, efficiency, and reasoning.

## Future Plans
- Add support for different difficulty levels of puzzles, as well as full puzzle solving.
- Integrate with popular LLM APIs for automated testing.
- Visualize results and performance metrics.

## Usage
I recommend familiarizing yourself with the package by running some initial base models to learn the CLI.

If you want to train just one model with one enhancement, use the following format:

```python src/main.py --model [MODEL_NAME] --enhancement [ENHANCEMENT]```

Any model which has a tokenizer in HuggingFace and weights in TransformerLens is compatible with this tool.

If you want to run a suite of models and enhancements one after another, I created a script ```run_all_models.sh``` which can be edited to include whichever models desired, as well as enhancements.
