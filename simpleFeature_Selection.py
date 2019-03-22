import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression, SelectFromModel
from sklearn.feature_selection import f_classif, mutual_info_classif, chi2
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


class feature_selection:
    """
     This select best features from the available features
    """
    def __init__(self, x, y):
        """
        :param x: Features
        :param y: Target
        """
        self.x = x
        self.y = y
        self.score__ = dict()
        self.model_support__ = dict()
        self.indices__ = np.arange(x.shape[-1])

    def plot_comparison(self):
        """
        :return: This will plot all models used for feature selection
        """
        plt.figure(figsize = (8, 8))
        indice_adj = 0
        for key, i in zip(self.score__.keys() , self.score__.values()):
            plt.bar(self.indices__ + indice_adj, i,width = 0.1, label = key)
            indice_adj += 0.25
        plt.legend()
        plt.title('Feature Selection')
        plt.xlabel('# of Features')
        plt.ylabel('Feature Score')
        plt.axis('tight')
        plt.xticks(self.indices__)
        plt.show()

    def ensemble_selection(self):
        """
        :return: This will combine all important features
        """
        feature_importance = np.array(list(self.model_support__.values()), dtype= np.int8).sum(axis = 0) / len(self.model_support__)
        return feature_importance >= feature_importance.mean()


class feature_selection_reg(feature_selection):
    """
    Feature Selection for Regression Models
    """
    def __init__(self, x, y):
        super().__init__(x,y)

    def f_reg(self):
        """
        F- test for regression, to capture univariate variable
        """
        model = SelectKBest(f_regression)
        model.fit(self.x, self.y)
        score = model.scores_  / model.scores_.max(axis = 0)
        self.score__['f_reg'] = score
        self.model_support__['f_reg'] = model.get_support()

    def f_lasso(self):
        """
        Linear LASSO to reduce unsignificant variables to zero
        """
        lasso = LassoCV(cv=5)
        lasso.fit(self.x, self.y)
        model = SelectFromModel(lasso)
        model.fit(self.x, self.y)
        score = np.abs(lasso.coef_ / lasso.coef_.max(axis = 0))
        self.model_support__['f_lasso'] = model.get_support()
        self.score__['f_lasso'] = score


    def f_mutInfo(self):
        """
        Mutual Info among variables
        """
        model = SelectKBest(mutual_info_regression)
        model.fit(self.x, self.y)
        score = model.scores_  / model.scores_.max(axis = 0)
        self.score__['f_mut_info'] = score
        self.model_support__['f_mut_info'] = model.get_support()

    def f_randForest(self):
        """
        Random forest with 20 trees parallel
        """
        rf = RandomForestRegressor(n_estimators= 20, random_state = 42)
        rf.fit(self.x, self.y)
        score = rf.feature_importances_ /  rf.feature_importances_.max(axis = 0)
        model = SelectFromModel(rf)
        model.fit(self.x, self.y)
        self.model_support__['f_RandForest'] = model.get_support()
        self.score__['f_RandForest'] = score


class feature_selection_class(feature_selection):
    """
    Feature Selection for classification
    """

    def __init__(self, x, y):
        super().__init__(x,y)

    def f_classif(self):
        model = SelectKBest(f_classif, k = 'all')
        model.fit(self.x, self.y)
        score = model.scores_  / model.scores_.max(axis = 0)
        self.score__['f_classif'] = score
        self.model_support__['f_classif'] = model.get_support()

    def f_LogReg(self):
        """
        Logistic Regression to select most important features
        """
        lr = LogisticRegressionCV(cv=5)
        lr.fit(self.x, self.y)
        model = SelectFromModel(lr)
        model.fit(self.x, self.y)
        score = np.abs(lr.coef_.sum(axis = 0))
        score /= score.max()
        self.model_support__['f_LogReg'] = model.get_support()
        self.score__['f_LogReg'] = score

    def f_mutInfo(self):
        """
        Mutual Information for classification
        """
        model = SelectKBest(mutual_info_classif, k = 'all')
        model.fit(self.x, self.y)
        score = model.scores_  / model.scores_.max(axis = 0)
        self.score__['f_mut_info'] = score
        self.model_support__['f_mut_info'] = model.get_support()

    def f_randForest(self):
        """
        Random forest for classification
        """
        rf = RandomForestClassifier(n_estimators= 50, random_state = 42)
        rf.fit(self.x, self.y)
        score = rf.feature_importances_ /  rf.feature_importances_.max(axis = 0)
        model = SelectFromModel(rf)
        model.fit(self.x, self.y)
        self.model_support__['f_RandForest'] = model.get_support()
        self.score__['f_RandForest'] = score

    def f_chi2(self):
        """
        Chi square test to select most significant features
        """
        model = SelectKBest(chi2, k = 'all')
        model.fit(self.x, self.y)
        score = model.scores_  / model.scores_.max(axis = 0)
        self.score__['f_chi2'] = score
        self.model_support__['f_chi2'] = model.get_support()


def featSelect_Regression(feat, target):
    """
    Automated Feature Selection for Regression Model
    :param feat: Training Features
    :param target: Training Target
    :return: list True or False for columns. True means imporatant and False means not important column
    """
    fs = feature_selection_reg(feat, target)
    fs.f_reg()
    fs.f_lasso()
    fs.f_mutInfo()
    fs.f_randForest()
    fs.plot_comparison()
    return fs.ensemble_selection()

