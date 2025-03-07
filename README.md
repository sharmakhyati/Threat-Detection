# Threat Detection API

This project provides a RESTful API using FastAPI to serve predictions from a pre-trained XGBoost classification model. The API accepts input data in JSON format and returns predicted classes for network attack categories.

## Features
- Load and use a pre-trained XGBoost model (`xgboost_model.model`).
- Accept input data in JSON format via an HTTP POST request.
- Preprocess input data, including label encoding of categorical features.
- Predict the class of the input data and return both integer and label predictions.
- Serve the API using FastAPI for quick and efficient inference.

## Prerequisites
- Python 3.8+
- Required libraries: `fastapi`, `uvicorn`, `pandas`, `xgboost`, `scikit-learn`, `joblib`

## Installation
```bash
pip install fastapi uvicorn pandas xgboost scikit-learn joblib
```

## How to Run
1. Start the FastAPI server using Uvicorn:
```bash
uvicorn main:app --reload
```

2. Access the interactive Swagger UI at:
```
http://localhost:8000/docs
```

## Example Request
```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{"id": 1, "dur": 0.000011, "proto": "udp", "service": "-", "state": "INT", "spkts": 2, ... }'
```

## Response Format
```json
{
  "predictions": [
    {"integer": 3, "label": "Normal"},
    {"integer": 1, "label": "DoS"}
  ]
}
```

## Project Structure
- `main.py`: FastAPI server with prediction endpoint.
- `xgboost_model.model`: Pre-trained XGBoost model.
- `label_encoder.pkl`: Label encoder for decoding prediction classes.

## License
This project is licensed under the MIT License.

