# PLTA-FinBERT: Pseudo-Label generation-based Test-Time Adaptation for Financial Sentiment Analysis

This repository contains the official implementation of PLTA-FinBERT, a test-time adaptation framework for financial sentiment analysis tasks (classification and regression). The framework dynamically adapts to semantic drift and scarce labeled data via pseudo-label generation and iterative model updates.
 

## Key Features
- **Test-Time Adaptation**: Models update parameters during inference using high-confidence pseudo-labels.
- **Multi-Perturbation Pseudo-Labeling**: Conventional NLP data augmentation (low-probability token replacement/insertion/deletion) ensures label reliability.
- **Task Agnostic**: Supports both sentiment classification and sentiment intensity regression.
- **Discriminative Fine-Tuning**: Selective parameter updates (encoder upper layers + classification/regression head) preserve pre-trained features.

## Experimental Results
| Task                  | Dataset                  | Key Metrics                  | PLTA-FinBERT Performance |
|-----------------------|--------------------------|------------------------------|--------------------------|
| Sentiment Classification | Financial Sentiment Analysis | Accuracy | 0.8288 |
| Sentiment Regression     | FiQA-SA                  | R² Score | 0.58 |


## Environment Setup
1. Clone this repository:
   ```bash
   git clone https://github.com/galaxywwww/PLTA-FinBERT.git
   cd PLTA-FinBERT

## Current Status: Coming Soon