# Spam Detection – Ready-to-Run Deployment

This pack is configured to use your **already trained model**: `spam_detection_model.joblib`.

## 🚀 Quick Start

1. Place your model file `spam_detection_model.joblib` in the same folder as these scripts.

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit UI:

```bash
streamlit run app_streamlit.py
```

4. Run the FastAPI service:

```bash
uvicorn app_fastapi:app --reload --port 8000
```

Test FastAPI with curl:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "Congratulations! You won a prize!"}'
```

Response example:
```json
{"label":"Spam","probability":0.987,"threshold":0.5}
```
