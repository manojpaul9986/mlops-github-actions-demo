from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier


def test_model_training():

    data = load_iris()

    model = RandomForestClassifier(
        n_estimators=10,
        random_state=42
    )

    model.fit(data.data, data.target)

    predictions = model.predict(data.data)

    assert len(predictions) == 150