# Iris ML API with GitHub Actions

An end-to-end MLOps practice project that trains a Random Forest classifier on scikit-learn's built-in Iris dataset, checks a minimum accuracy threshold, serves predictions through FastAPI, and runs tests and model training in GitHub Actions. The workflow also builds and publishes a Docker image on pushes to `main`.

## Project Structure

```text
.
|-- .github/workflows/ml_pipeline.yml  # Test, train, save artifact, build and push image
|-- src/
|   |-- train.py                       # Train, evaluate, and save the model
|   `-- app.py                         # FastAPI prediction service
|-- tests/
|   |-- test_data.py                   # Iris dataset checks
|   `-- test_model.py                  # Model training smoke test
|-- Dockerfile
|-- requirements.txt
`-- README.md
```

Training creates `artifacts/model.pkl`. This generated file is ignored by Git, so create it before starting the API or building the Docker image.

## Requirements

- Python 3.11
- pip
- Docker (optional, for containerized use)

## Setup

Run these commands from the project root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For macOS or Linux, activate the environment with `source .venv/bin/activate` instead.

## Run Tests and Train

```powershell
pytest
python src/train.py
```

The training script uses an 80/20 stratified train/test split and a 150-tree Random Forest. It prints test accuracy and stops with an error if accuracy is below `0.85`. On success, it writes `artifacts/model.pkl`.

## Run the API

After training, start the service from the project root:

```powershell
uvicorn src.app:app --reload
```

The service is available at `http://127.0.0.1:8000`. Interactive API documentation is at `http://127.0.0.1:8000/docs`.

### Make a Prediction

Send four measurements in centimeters:

```powershell
Invoke-RestMethod -Method Post `
	-Uri http://127.0.0.1:8000/predict `
	-ContentType 'application/json' `
	-Body '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
```

Example response:

```json
{"prediction":"setosa"}
```

The API accepts `sepal_length`, `sepal_width`, `petal_length`, and `petal_width` as numeric fields. Predictions are one of `setosa`, `versicolor`, or `virginica`.

## Build and Run with Docker

Train the model first, then build and start the container from the project root:

```powershell
python src/train.py
docker build -t iris-ml:demo .
docker run --rm -p 8000:8000 iris-ml:demo
```

Open `http://127.0.0.1:8000/docs` to try the API. The container listens on port `8000`.

## GitHub Actions

The workflow at `.github/workflows/ml_pipeline.yml` runs on pushes and pull requests targeting `main`. It installs dependencies, runs `pytest`, trains the model, and uploads `artifacts/model.pkl` as the `trained-model` workflow artifact. For pushes to `main`, it also builds and pushes the `iris-ml:v1` image to Docker Hub.

To enable the Docker Hub steps, add these repository Actions secrets in GitHub:

- `DOCKERHUB_USERNAME`: Docker Hub username
- `DOCKERHUB_TOKEN`: Docker Hub access token

Without these secrets, tests and training can still run, but the Docker login, build, and push steps will not succeed on a `main` push.

## Sharing This Project

For a demo ZIP, include the source, tests, Dockerfile, dependency list, README, license, and `.github/workflows/ml_pipeline.yml`. For an immediately runnable handoff, run `python src/train.py` and include the generated `artifacts/model.pkl` as well; it is ignored by Git but required by the API and Docker build. If the recipient can install the dependencies and run training, the model artifact can be left out.

Do not include `.git/`, `.venv/` or `venv/`, `__pycache__/`, `.pytest_cache/`, or machine-specific environment files. Never put credentials or Docker Hub tokens in the ZIP. If the GitHub Actions workflow is not part of the demonstration, `.github/` can be omitted too.