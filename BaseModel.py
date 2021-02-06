import pandas as pd
import numpy as np
import pickle
from sklearn.compose import make_column_transformer
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import category_encoders as ca
from tensorflow import keras
import datetime as dt

from constants import TARGET,NOT_FEATURES
from feature_generator import configure_df,get_last_team_game,get_past_wins_against

class BaseModel():
    def __init__(self,model_name,df,teams_df):
        self.model_name = model_name
        self.df = df
        self.teams_df = teams_df
        self.cat_encoding = ca.CatBoostEncoder()
        self.scaler = StandardScaler()
        self.clf = None
        self.save_folder = 'saved_models'

        self.features =  list(filter(lambda x: x not in NOT_FEATURES,df.columns ))
        self.category_columns = df[self.features].select_dtypes('category').columns
        self.to_standard_features = df[self.features].select_dtypes([int,float]).columns

        self.column_transformer = make_column_transformer( 
            (self.cat_encoding,self.features),
            (self.scaler,self.to_standard_features),
            remainder='passthrough') 

    def load_model(self):
        self.column_transformer = pickle.load(open(f'saved_objects/column_transformer_{self.model_name}.pkl', 'rb'))
        print("Model loaded")

    def train(self,X_train,y_train):
        print("Preparing the Training data")
        # Encoding the Target
        y_train_encoded = y_train
        # Column tranformer
        self.column_transformer.fit(X_train,y_train_encoded)
        pickle.dump(self.column_transformer, open(f"saved_objects/column_transformer_{self.model_name}.pkl", 'wb'))
        # Transform the features
        X_train_encoded = self.column_transformer.transform(X_train)
        class_weights = compute_class_weight('balanced', np.unique(y_train_encoded) , y_train_encoded)
        self.class_weights = dict(enumerate(class_weights))
        return X_train_encoded, y_train_encoded

    def evaluate(self,X_test,y_test):
        X_test_encoded = self.column_transformer.transform(X_test)
        y_test_encoded = y_test
        y_pred = self.clf.predict(X_test_encoded)
        return classification_report(y_pred.round(),y_test_encoded)

    def transformX(self,X):
        return self.column_transformer.transform(X)

    def generateXMounted(self,X_test):
        X_test_mounted = pd.DataFrame()
        for row in X_test.itertuples():
            X_test_mounted = X_test_mounted.append(self.create_prediction_df(row.blue,row.red,self.df.loc[row.Index].date)[self.features],ignore_index=True)
            #X_test_mounted_encoded = self.column_transformer.transform(X_test_mounted)
        return X_test_mounted


    def create_prediction_df(self,blue,red,date_cut=dt.datetime.now()):
        past_df = self.df[self.df.date <=  date_cut]
        start_data = [
            {'blue':blue,
            'red':red,
            'Date':(dt.datetime.now() - dt.timedelta(hours=24)).strftime('%Y-%m-%d'),
            'Score': '0 - 0'
            }
        ]
        prediction_df = pd.DataFrame(start_data)
        prediction_df = configure_df(prediction_df)

        blue_last_game = get_last_team_game(blue,past_df)
        red_last_game = get_last_team_game(red,past_df)
        remaining_columns = np.setdiff1d(prediction_df.columns,self.features)

        for column in self.features:
            if 'blue' in column.lower() and column != 'blue':
                last_game_column_name = column.lower().replace('blue_','').replace('blue','')
                prediction_df[column] = blue_last_game[last_game_column_name]
                continue
            if 'red' in column.lower() and column != 'red':
                last_game_column_name = column.lower().replace('red_','').replace('red','')
                prediction_df[column] = red_last_game[last_game_column_name]
                continue

        prediction_df['blue_winst_against'] = prediction_df.apply(lambda x: get_past_wins_against(x,'blue',self.df),axis=1)
        prediction_df['red_winst_against'] = prediction_df.apply(lambda x: get_past_wins_against(x,'red',self.df),axis=1)

        return prediction_df


    def prepare_prediction(self,blue,red,date_cut=dt.datetime.now()):
        prediction_df = self.create_prediction_df(blue,red,date_cut)
        X_prediction = self.column_transformer.transform(prediction_df[self.features])
        return X_prediction


    def predict(self,blue,red):
        to_predict = self.prepare_prediction(blue,red)
        prediction = self.clf.predict(to_predict)[0]

        rounded_prediction = prediction.round()

        if rounded_prediction ==1:
            return f"{blue} Will Win!",prediction
        elif rounded_prediction == 0:
            return f"{red} Will Win!",prediction
        else :
            return f"Error predicting",prediction