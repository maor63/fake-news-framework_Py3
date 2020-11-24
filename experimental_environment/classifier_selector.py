from commons.consts import Classifiers
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import xgboost as xgb
from sklearn import tree

class Classifier_Selector():
    def __init__(self):
        self._selected_classifier = None

    def get_classifier_by_type(self, classifier_type_name):

        if classifier_type_name == Classifiers.RandomForest:
            self._selected_classifier = RandomForestClassifier(n_estimators=100)

        elif classifier_type_name == Classifiers.DecisionTree:
            self._selected_classifier = tree.DecisionTreeClassifier()

        elif classifier_type_name == Classifiers.AdaBoost:
            self._selected_classifier = AdaBoostClassifier(n_estimators=30)

        elif classifier_type_name == Classifiers.XGBoost:
            self._selected_classifier = xgb.XGBClassifier()

        return self._selected_classifier
