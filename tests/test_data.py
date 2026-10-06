from sklearn.datasets import load_iris


def test_iris_dataset():

    data = load_iris()

    assert data.data.shape == (150, 4)

    assert len(data.target) == 150