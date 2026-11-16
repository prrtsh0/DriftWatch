FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# train a model first on your machine (python train_service_model.py) so ./artifacts exists
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
