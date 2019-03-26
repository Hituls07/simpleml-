import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import  precision_score, recall_score, accuracy_score
from sklearn.base import clone
# from sklearn.svm import LinearSVC, SVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.datasets import load_wine, load_iris, load_breast_cancer



class ComparisonModels:

    def __init__(self, feat, target, *args, **kwargs):
        """
        To compare different models (on the same data), in terms of accuracy,  recall, error rate, and time
        :param feat: Features of training data
        :param target: target of training data
        :param kwargs: to handle unspecified parameters
        """
        self.feat = feat
        self.target = target
        self.accuracy__, self.precision__, self.recall__, self.timeit__, self.error_rate__ = {}, {}, {}, {}, {}

    def train_model(self, classifiers):

        """
        To train models on input classifiers
        :param classifiers: classifiers to compare
        :return: self
        """
        x_train, x_test, y_train, y_test = train_test_split(self.feat, self.target, test_size = 0.2, random_state = 42)

        for name, clf in classifiers:
            t_start = time.time()
            model = clone(clf)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            self.accuracy__[name] = accuracy_score(y_test, y_pred)
            t_end = time.time()
            #  Precision and Recall are eligible for only binary classifiers
            if np.unique(y_train).shape[0] <= 2:
                self.precision__[name] = precision_score(y_test, y_pred)
                self.recall__[name] = recall_score(y_test, y_pred)
            self.timeit__[name] = t_end - t_start
        self.error_rate__ = dict(list(map(lambda x: (x, 1 - self.accuracy__[x]), self.accuracy__)))
        return self

    def error_rate_plot(self):
        """
        Error plot comparison
        :return: Plot comparing error rates
        """
        plt.bar(self.error_rate__.keys(), self.error_rate__.values())
        plt.title('Error Rates Comparison')
        plt.xlabel('Models')
        plt.ylabel('Error Rate')
        plt.axis('tight')
        plt.ylim([0, 1])
        plt.show()

    def time_rate_plot(self):
        """
        Time rate plot comparison
        :return: Plot comparing time to run models
        """
        plt.bar(self.timeit__.keys(), self.timeit__.values())
        plt.title('Time Rate Comparison')
        plt.xlabel('Models')
        plt.ylabel('Time Rate')
        plt.axis('tight')
        plt.ylim([0, 1])
        plt.show()

    def accuracy_plot(self):
        """
        Accuracy plot comparison
        :return: Plot comparing accuracies of models
        """
        plt.bar(self.accuracy__.keys(), self.accuracy__.values())
        plt.title('Accuracies Comparison')
        plt.xlabel('Models')
        plt.ylabel('Accuracy')
        plt.axis('tight')
        plt.ylim([0, 1])
        plt.show()

    def precision_plot(self):
        """
        Precision plot comparison
        :return:  Plot comparing precison of models
        """
        plt.bar(self.accuracy__.keys(), self.accuracy__.values())
        plt.title('Precisions Comparison')
        plt.xlabel('Models')
        plt.ylabel('Precision')
        plt.axis('tight')
        plt.ylim([0, 1])
        plt.show()

    def recall_plot(self):
        """
        Recall plot comparison
        :return:  Plot comparing Recall of models
        """
        plt.bar(self.recall__.keys(), self.recall__.values())
        plt.title('Recall Comparison')
        plt.xlabel('Models')
        plt.ylabel('Recall')
        plt.axis('tight')
        plt.ylim([0, 1])
        plt.show()



# bcs = load_breast_cancer()
# bcs_feat = bcs['data']
# bcs_tar = bcs['target']
# bcs_clf = [
#         ('Decision Trees', DecisionTreeClassifier(max_depth= 2)),
#         ('Logistic Regression', LogisticRegression())
#                             ]
#
# c = ComparisonModels(bcs_feat, bcs_tar)
# c.train_model(bcs_clf)
# c.accuracy_plot()
# c.recall_plot()



