import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def plotLearningCurve(estimator, feat, tar, classification=True):

    """
    :param estimator:  estimator model or pipeline
    :param feat: training features
    :param tar: training target
    :param classification: true = classification, false = regression
    :return: learning curve of model to examine model's behaviour
    """

    train_feat, val_feat, train_tar, val_tar = train_test_split(feat, tar, test_size=0.2)  # Split into validation
    m = len(train_tar)
    train_score, val_score = [], []
    points = list(np.linspace(3, m, 10, dtype=np.int64))

    for row in points:
        estimator.fit(train_feat[:row], train_tar[:row])  # fitting model here
        train_predict = estimator.predict(train_feat[:row])
        val_predict = estimator.predict(sval_feat)
        if classification:
            # Training Accuracy
            train_accuracy = accuracy_score(train_tar[:row], train_predict)
            train_score.append(train_accuracy)
            # Validation Accuracy
            val_accuracy = accuracy_score(val_tar, val_predict)
            val_score.append(val_accuracy)
        else:
            # Training RMSE
            train_accuracy = mean_squared_error(train_tar[:row], train_predict)
            train_score.append(train_accuracy)
            # Validation RMSE
            val_accuracy = mean_squared_error(val_tar, val_predict)
            val_score.append(val_accuracy)

    # Plotting scores
    plt.figure(figsize=(8, 8))
    plt.plot(points, train_score, 'r-', marker='o', label='Training Accuracy')
    plt.plot(points, val_score, 'b-', marker='x', label='Validation Accuracy')
    plt.legend()
