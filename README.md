# Temporal Transformer for Financial Volatility Forecasting

A PyTorch-based hybrid deep learning framework that leverages Multi-Head Attention Encoders and Sinusoidal Positional Encodings to forecast short-term market volatility. The model is evaluated against a rolling 1-step-ahead statistical baseline across out-of-sample financial return series.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Highlights

* **Leak-Free Normalization**: Features (`StandardScaler`) are fit strictly on the training partition to prevent lookahead bias.
* **Temporal Encodings**: Incorporates custom `PositionalEncoding` layers to preserve time-series sequence order inside the self-attention block.
* **Rigorous Baseline Evaluation**: Evaluates out-of-sample predictions against a rolling 1-step-ahead historical volatility baseline.

---

## Model Architecture

The core pipeline transforms multi-dimensional time-series features (log returns and 5-day rolling realized volatility) through a multi-layer Transformer Encoder:

```mermaid
graph LR
    A["Input Features X ∈ ℝ^(B×L×D_in)"] -->|Linear Projection| B["Projection ℝ^(B×L×D_model)"]
    B -->|+ Positional Encoding| C["Transformer Encoder"]
    C -->|Extract Last Step t=L| D["Linear Output Head"]
    D --> E["Prediction ŷ"]

1. **Input Projection Layer**: Projects sequence inputs to $D_{\text{model}} = 64$.
2. **Positional Encoding**: Sinusoidal positional embeddings preserve time order.
3. **Transformer Encoder**: 2-layer, 4-head attention block with dropout ($p = 0.1$).
4. **Output Head**: Regression head operating on the final sequence step $T$ to predict 1-step-ahead volatility.
```

---

## Quickstart

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/YogeshRajasekhar/Financial-Volatility-Forecasting-via-Temporal-Transformers.git
cd Financial-Volatility-Forecasting-via-Temporal-Transformers
pip install -r requirements.txt
