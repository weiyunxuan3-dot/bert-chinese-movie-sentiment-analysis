# Chinese Movie Review Sentiment Classification Using BERT

A course final project that applies a fine-tuned BERT model to classify Chinese movie reviews into positive and negative sentiment categories.

## Technologies
Python, PyTorch, Hugging Face Transformers, Pandas, Jieba, Scikit-learn, Gradio

## Project Workflow
1. Data cleaning and sentiment labelling
2. Balanced sampling and Easy Data Augmentation (EDA)
3. Fine-tuning a Chinese BERT model
4. Model evaluation using Accuracy, Precision, Recall and F1-score
5. Interactive sentiment prediction with Gradio

## Results
- Training samples after augmentation: 80,000
- Validation samples: 5,000
- Test samples: 5,000
- Test accuracy: **87.50%**
- Precision: **86.62%**
- Recall: **88.80%**
- F1-score: **0.8769**

## Project Files
- `Data processing.py` — Data cleaning, sentiment labelling, balanced sampling, data augmentation and word cloud generation
- `BERT_Optimized.ipynb` — BERT model training, evaluation and interactive sentiment prediction
- `synonyms.json` — Synonym dictionary used for data augmentation

## Project Context
Course Final Project, June 2026.

This project demonstrates the application of natural language processing and deep learning techniques to sentiment analysis of user-generated movie reviews.
