FROM python:3.11-slim
WORKDIR /app

# Where the model + reference sample live. Set as an env var so the build-time
# training step and the runtime app read the SAME path (app.py + train script).
ENV DRIFTWATCH_ARTIFACTS=artifacts

# Training flags for the self-contained artifact generated at build time.
# Default: synthetic demo data (no internet needed, reliable on Render).
# For a real model, deploy with build arg: --build-arg TRAIN_FLAGS="--state CA"
ARG TRAIN_FLAGS="--synthetic --n-max 20000"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# artifacts/ is gitignored, so generate the model + reference sample here so the
# service is self-contained (otherwise /predict and /drift return 503).
RUN python train_service_model.py $TRAIN_FLAGS

EXPOSE 8000
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
