# Sentiment Analysis: Deep Learning vs LLMs

This project compares **traditional deep learning models** and **Large Language Models (LLMs)** for **sentiment analysis** using the **Amazon Fine Food Reviews** dataset. The goal is to provide a fair, reproducible, and practical comparison between training task-specific models and leveraging general-purpose LLMs via prompting.

---

## 📌 Objectives

* Perform sentiment analysis on product reviews
* Compare **Deep Learning** vs **LLM-based** approaches
* Evaluate performance, cost, complexity, and deployment trade-offs
* Analyze when each approach is preferable

---

## 📂 Dataset

**Amazon Fine Food Reviews**

Each review contains both structured metadata and free-text fields.

Main columns used:

* `Score` – Rating from 1 to 5
* `Summary` – Short review title
* `Text` – Full review text

### Sentiment Labeling

We use a **binary classification** setup:

| Score | Sentiment |
| ----- | --------- |
| ≥ 4   | Positive  |
| ≤ 2   | Negative  |
| 3     | Removed   |

This strategy reduces label noise and is commonly used in the literature.

---

## 🧠 Approaches Compared

### 1️⃣ Deep Learning Models

These models are trained or fine-tuned on the dataset.

#### Models

* **BiLSTM + Pretrained Embeddings (GloVe)**
* **TextCNN**
* **Fine-tuned BERT (`bert-base-uncased`)**

#### Architecture Example (BiLSTM)

```
Embedding → BiLSTM → Global Max Pooling → Dense → Sigmoid
```

#### Characteristics

* Requires labeled training data
* Higher upfront training cost
* Low inference latency
* Suitable for production deployment

---

### 2️⃣ LLM-based Sentiment Analysis

Large Language Models are used **without fine-tuning**, relying on **prompt engineering**.

#### Prompting Strategies

* **Zero-shot**: No examples provided
* **Few-shot**: 2–4 labeled examples included in the prompt

Example prompt:

```
Classify the sentiment of the following review as Positive or Negative.

Review:
"This coffee tastes amazing and fresh."

Answer with only one word: Positive or Negative.
```

#### Characteristics

* No training required
* High inference cost
* Sensitive to prompt design
* Strong generalization and reasoning

---

## 📊 Evaluation Metrics

All models are evaluated using the same metrics:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion Matrix

LLM inference is performed with deterministic settings (e.g., temperature = 0).

---

## ⚖️ Comparison Dimensions

| Aspect            | Deep Learning | LLMs          |
| ----------------- | ------------- | ------------- |
| Training cost     | High          | None          |
| Inference cost    | Low           | High          |
| Latency           | Low           | High          |
| Data requirement  | Large         | Small         |
| Domain adaptation | Excellent     | Good          |
| Interpretability  | Medium        | High          |
| Deployment        | Easy          | API-dependent |

---

## 📈 Expected Outcomes

* Fine-tuned BERT typically outperforms LSTM and CNN models
* Few-shot LLMs can match or exceed fine-tuned models on small samples
* Zero-shot LLMs may underperform on domain-specific language
* LLMs handle long and nuanced reviews better

---

## 🧪 Experimental Protocol

* Fixed random seeds
* Same train/test split for all models
* Identical label definitions
* Same evaluation metrics
* Same test subset used for LLM inference

---

## 📁 Project Structure

```
├── data/
├── preprocessing/
├── models/
│   ├── lstm.py
│   ├── cnn.py
│   ├── bert.py
│   └── llm_inference.py
├── evaluation/
├── notebooks/
└── README.md
```

---

## 🔍 Key Research Questions

* How much labeled data is required for each approach?
* Is prompt engineering a viable alternative to fine-tuning?
* How stable are LLM predictions across runs?
* Does domain specificity favor trained models?
* What is the cost–performance trade-off?

---

## 🚀 Future Extensions

* Multiclass sentiment classification
* Error analysis on model disagreements
* Calibration analysis
* Cost-per-inference comparison
* Human evaluation subset

---

## 📜 License

This project is intended for educational and research purposes.
