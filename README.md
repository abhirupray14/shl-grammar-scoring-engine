# SHL Grammar Scoring Engine

A machine learning solution for predicting English grammar proficiency scores from spoken audio recordings. Developed for the **SHL Hiring Assessment 2026** challenge.

## Overview

The objective is to predict a continuous grammar score between **0 and 5** from audio recordings approximately 45–60 seconds long.

The solution combines automatic speech recognition, linguistic feature extraction, semantic text embeddings, and regression models to estimate the grammar quality of spoken English.

## Problem Statement

Given an audio recording of a speaker, predict its corresponding grammar assessment score. The training dataset contains audio recordings paired with human-assigned scores, while the test dataset contains recordings for which predictions must be generated.

The dataset is provided through the private Kaggle competition and is not included in this repository.

## Solution Architecture

```text
             Audio Recording
                    |
                    v
          Audio Preprocessing
                    |
                    v
          Whisper Transcription
                    |
                    v
              Transcript
                    |
          +---------+----------+
          |                    |
          v                    v
    spaCy Linguistic      MPNet Sentence
    Feature Extraction    Embeddings
          |                    |
          +---------+----------+
                    |
                    v
           Feature Combination
                    |
                    v
          LightGBM Regression
                    |
                    v
          Grammar Score (0–5)
                    |
                    v
          Submission CSV File
```

## Methodology

### 1. Speech-to-Text Transcription

Whisper is used to convert audio recordings into text transcripts. The transcripts provide the linguistic information required for downstream grammar scoring.

### 2. Linguistic Feature Extraction

spaCy is used to extract linguistic and syntactic characteristics from the transcripts, including part-of-speech information and sentence structure features.

### 3. Sentence Embeddings

The `all-mpnet-base-v2` model from Sentence Transformers generates dense vector representations of the transcripts. These embeddings provide contextual information about the text.

### 4. Grammar Score Prediction

LightGBM regression is trained using the extracted linguistic features and sentence embeddings to predict continuous grammar scores.

Ridge regression with TF-IDF text features was also explored as an alternative modeling approach and for ensemble experiments.

### 5. Validation and Evaluation

Five-fold cross-validation is used to assess model generalization. The notebook evaluates predictions using:

- **Root Mean Squared Error (RMSE):** Measures the magnitude of prediction errors.
- **Pearson Correlation:** Measures the linear correlation between predicted and actual scores.

The final predictions are constrained to the valid score range of 0–5.

## Experimental Results

The following results were obtained for the MPNet + LightGBM experimental configuration:

| Metric | Result |
|---|---:|
| Five-fold OOF RMSE | 0.9217 |
| Five-fold OOF Pearson Correlation | 0.6742 |
| Training RMSE | 0.2023 |
| Training Pearson Correlation | 0.9907 |

**Note:** Out-of-fold (OOF) metrics provide a more useful estimate of generalization than training metrics. The gap between training and OOF performance indicates that overfitting remains a consideration.

## Technology Stack

- **Language:** Python
- **Speech Recognition:** OpenAI Whisper
- **NLP:** spaCy
- **Text Embeddings:** Sentence Transformers (`all-mpnet-base-v2`)
- **Machine Learning:** LightGBM, scikit-learn
- **Text Features:** TF-IDF
- **Data Processing:** NumPy, Pandas
- **Visualization:** Matplotlib, Seaborn
- **Development Environment:** Kaggle Notebooks

## Repository Structure

```text
shl-grammar-scoring-engine/
├── shl_grammar_scoring.ipynb
├── README.md
└── requirements.txt
```

The notebook contains the implementation, model evaluation, visualizations, and submission generation workflow.

## Getting Started

### Prerequisites

Python 3.10 or a compatible environment with the required dependencies.

### Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/shl-grammar-scoring-engine.git
cd shl-grammar-scoring-engine
```

Install the dependencies:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

The pretrained Whisper and Sentence Transformers models may require downloading their model weights on first use.

### Running the Notebook

1. Open `shl_grammar_scoring.ipynb` in Kaggle or a compatible Jupyter environment.
2. Configure access to the competition dataset.
3. Run the notebook cells in sequence.
4. Generate predictions for the test recordings.
5. Save the final predictions as `submission.csv`.

The notebook may require adjustments to dataset paths and compute settings when run outside Kaggle.

## Output Format

The generated submission file contains two columns:

| Column | Description |
|---|---|
| `filename` | Name of the test audio file |
| `label` | Predicted grammar score between 0 and 5 |

Example:

```csv
filename,label
audio_001.wav,3.42
audio_002.wav,4.15
```

The values above are illustrative examples, not actual competition predictions.

## Future Improvements

Potential improvements include:

- More detailed grammatical error and dependency features.
- Comparison of additional pretrained text embedding models.
- Regularization and hyperparameter tuning.
- Complementary regression models and out-of-fold ensembling.
- Acoustic and prosodic features such as speech rate and pause statistics.
- Pretrained speech representations from models such as WavLM or HuBERT.

Each improvement should be evaluated using consistent cross-validation splits and both RMSE and Pearson correlation.

## Reproducibility and Data Usage

The competition dataset is not redistributed in this repository. Access to the audio files and labels must follow the competition's access rules.

Model performance can vary with the validation split, model configuration, dependencies, and pretrained model versions.

## Acknowledgements

Developed as part of the SHL Hiring Assessment 2026 private Kaggle challenge for the Research Engineer hiring process.

