import streamlit as st
import datetime as dt
import pandas as pd
import category_encoders as ca

from sklearn.compose import make_column_transformer
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from feature_generator import set_all_features,get_teams_df
from NNModel import NNModel
from constants import TARGET


st.set_page_config(layout="wide")

@st.cache
def getGames(years=list(range(2015,dt.datetime.now().year +1,1)),link="https://gol.gg/tournament/tournament-matchlist/LPL%20Spring%20{}/"):
  df = pd.DataFrame()
  for year in years:
    result = pd.read_html(link.replace('{}',str(year)))
    df = df.append(result,ignore_index=True)
  return df

@st.cache
def getXTestMounted(X_test):
    return nn.generateXMounted(X_test)

 

games = getGames()
df = set_all_features(games)
teams_df = get_teams_df(df)

backtest_df = df[-10:]
df = df[:-10]

# Models
nn = NNModel('NN',df,teams_df)

# SPLIT X & Y
X = df[:-10][nn.features].copy()
y = df[:-10][TARGET].copy()
# Split and encode train & test
X_train, X_test,y_train,y_test  = train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)
# ====================== NN Model =================
nn.load_model()
#nn.train(X_train,y_train,X_test,y_test)


X_test_mounted = getXTestMounted(X_test)

#Evaluations
nn_evaluation = nn.evaluate(X_test,y_test)
nn_mounted_evaluation = nn.evaluate(X_test_mounted,y_test)


# ================================ LAYOUT ================================
st.header("Predict LPL")
model_stats_left,model_stats_right = st.beta_columns((1,1))
model_stats_left.subheader('NN')
model_stats_left.text(nn_evaluation)
model_stats_right.subheader('NN Mounted')
model_stats_right.text(nn_mounted_evaluation)

selection_left, selection_right = st.beta_columns((1,1))
blue = selection_left.selectbox("Blue",options=teams_df['team'].values)
red = selection_right.selectbox("Red",options=teams_df['team'].values)

#X_prediction = nn.prepare_prediction(blue,red)
nn_prediction,nn_probabilitys = nn.predict(blue,red)

selection_left.table(teams_df[teams_df.team == blue].astype('str'))
selection_right.table(teams_df[teams_df.team == red].astype('str'))

result_left,result_center,result_right =  st.beta_columns(3)

result_left.success(nn_prediction)
result_right.text(nn_probabilitys)