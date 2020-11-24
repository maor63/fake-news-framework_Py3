# Created by Aviad Elyashar  12/07/2017
from .predictor import Predictor
import pandas as pd
import io, json

class Clickbait_Challenge_Predictor(Predictor):
    def __init__(self, db):
        Predictor.__init__(self, db)

        self._json_results_file = self._config_parser.eval(self.__class__.__name__, "json_results_file")

    def _write_predictions_into_file(self, classifier_type_name, num_of_features,
                                     unlabeled_index_field_series, predictions_series, predictions_proba_series):

        for targeted_class_field_name in self._targeted_class_field_names:
            unlabeled_dataframe_with_prediction = pd.DataFrame(unlabeled_index_field_series,
                                                               columns=[self._indentifier_field_name])

            unlabeled_dataframe_with_prediction.columns = ['id']

            unlabeled_dataframe_with_prediction.reset_index(drop=True, inplace=True)
            unlabeled_dataframe_with_prediction["clickbaitScore"] = predictions_proba_series

            unlabeled_dataframe_with_prediction_json = unlabeled_dataframe_with_prediction.to_json(orient='records', lines=True)

            #unlabeled_dataframe_with_prediction_json = str(unlabeled_dataframe_with_prediction_json)



            with io.open(self._json_results_file, 'w', encoding='utf-8') as file:
                file.write(json.dumps(unlabeled_dataframe_with_prediction_json, ensure_ascii=False))


            #with open(self._json_results_file, 'w') as outfile:
            #    json.dump(unlabeled_dataframe_with_prediction_json, outfile, encoding='ascii')



