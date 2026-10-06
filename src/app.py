from fastapi import FastAPI
from pydantic import BaseModel
import joblib


app = FastAPI(
    title="Iris Classification API"
)


# Load trained model
model = joblib.load(
    "model.pkl"
)


class IrisInput(BaseModel):

    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


class PredictionOutput(BaseModel):

    prediction: str


class_names = [
    "setosa",
    "versicolor",
    "virginica"
]


@app.get("/")
def home():

    return {
        "message": "Iris ML model is running"
    }


@app.post("/predict")
def predict(data: IrisInput):

    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    prediction = model.predict(features)[0]

    return {
        "prediction": class_names[prediction]
    }