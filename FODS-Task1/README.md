# Industry Classification

## Files

```text
submission/
├── df_train.pkl
├── df_financials_train.pkl
├── requirements.txt
├── README.md
└── solution.ipynb
```

## Setup

Use Python 3.10+ and install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

```python
from solution import train, predict

model = train(df_train, df_fin_train)
y_pred, y_confidence = predict(model, df_test, df_fin_test)
```

- `y_pred`: integer-encoded industry predictions.
- `y_confidence`: confidence estimates between 0 and 1.
- Decode labels with `model["label_encoder"].inverse_transform(y_pred)`.

Training includes five-fold out-of-fold fitting and took approximately 25 minutes on the development machine.

## Validation

Using a 90/10 split (`random_state=42`), the implementation achieved:

| Metric | Result |
|---|---:|
| Accuracy | 0.7915 |
| Weighted F1 | 0.7889 |
| Brier score | 0.1270 |
| Final score | 0.6887 |
