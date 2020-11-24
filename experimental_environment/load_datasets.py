# Created by jorgeaug at 16/02/2017
from preprocessing_tools.abstract_controller import AbstractController
import pandas as pd
from sklearn import preprocessing
import uuid

class Load_Datasets(AbstractController):
    def __init__(self, db):
        AbstractController.__init__(self, db)

    def set_up(self):
        pass

    def execute(self, window_start=None):
        self._db.delete_authors_features()
        df = pd.read_csv('C:/Users/jorgeaug/Desktop/keel_datasets/automobile/automobile.csv')
        categorical_columns = ['Make', 'Fuel-type', 'Aspiration', 'Num-of-doors', 'Body-style', 'Drive-wheels',
                              'Engine-location', 'Engine-type', 'Num-of-cylinders', 'Fuel-system']
        le = preprocessing.LabelEncoder()
        for col in categorical_columns:
            df[col] = le.fit_transform(df[col])
        columns = list(df)
        print(columns)
        authors_features = []
        for row_index, row in df.iterrows():
            author_guid = str(uuid.uuid4())
            for col in columns:
                attribute_name = str(col)
                attribute_value = row[col]
                author_feature = self._db.create_author_feature(author_guid, attribute_name, attribute_value)
                authors_features.append(author_feature)
        if authors_features:
            for author_features_row in authors_features:
                self._db.update_author_features(author_features_row)
            self._db.commit()