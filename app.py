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
import requests


st.set_page_config(layout="wide")

@st.cache
def getGames(years=list(range(2015,dt.datetime.now().year +1,1)),link="https://gol.gg/tournament/tournament-matchlist/LPL%20Spring%20{}/"):
  df = pd.DataFrame()
  for year in years:
    final_link = link.replace('{}',str(year))
    result = pd.read_html(requests.get(final_link).text)
    df = df.append(result,ignore_index=True)
  return df

@st.cache
def get_all_games(links,years=list(range(2015,dt.datetime.now().year +1,1))):
  df = pd.DataFrame()
  for link in links.values():
    for year in years:
      final_link = link.replace('{}',str(year))
      result = pd.read_html(requests.get(final_link).text)
      df = df.append(result,ignore_index=True)
  return df

@st.cache
def getXTestMounted(X_test):
    return nn.generateXMounted(X_test)


def runBackTest(model,backtest_df):
    corrects = 0
    of = len(backtest_df)
    for item in backtest_df.itertuples():
        match = backtest_df.loc[item.Index]
        to_pred_m = model.prepare_prediction(match.blue,match.red)
        model_pred = model.predict(match.blue,match.red)[1][0].round()
        if model_pred == item.blue_is_winner:
            corrects +=1

    return (f"{corrects} / {of} on backtest of {model.model_name}")

def predictNextGames(next_games):
    next_games['Prediction'] = next_games.apply(lambda x: nn.predict(x['Blue Side'],x['Red Side']),axis=1 )
    return next_games


GAME_LINKS =  { 
  "LPL":"https://gol.gg/tournament/tournament-matchlist/LPL%20Spring%20{}/",
  "LCK":"https://gol.gg/tournament/tournament-matchlist/LCK%20Spring%20{}/",
  "CBLOL1":"https://gol.gg/tournament/tournament-matchlist/CBLOL%20Split%201%20{}/",
  "CBLOL2":"https://gol.gg/tournament/tournament-matchlist/CBLOL%20Split%202%20{}/"
}


# ============================= SIDEBAR ===========================
st.sidebar.subheader("Treinar")
#tournament = st.sidebar.selectbox("Tournament",list(GAME_LINKS.keys()))
# ================================================================
games =  get_all_games(GAME_LINKS) #getGames(link=GAME_LINKS[tournament])
df = set_all_features(games)
next_games = games[games.Score == '-']
teams_df = get_teams_df(df)

backtest_df = df[-20:]
df = df[:-20]

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
backtest_result = runBackTest(nn,backtest_df)

next_games = predictNextGames(next_games)

# ================================ LAYOUT ================================


treinar_nn = st.sidebar.button("Treinar NN")

if treinar_nn:
  nn.train(X_train,y_train,X_test,y_test)

st.header("Predict LPL")
model_stats_left,model_stats_center,model_stats_right = st.beta_columns((1,1,1))
model_stats_left.subheader('NN')
model_stats_left.text(nn_evaluation)
model_stats_center.subheader('NN Mounted')
model_stats_center.text(nn_mounted_evaluation)
model_stats_right.subheader("Back Tests")
model_stats_right.text(backtest_result)

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

st.header("Next Games")
st.dataframe(next_games)

df.to_csv('lpl.csv')