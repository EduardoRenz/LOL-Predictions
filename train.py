#%%
import datetime as dt
import pandas as pd
from feature_generator import set_all_features,get_teams_df
# %%
def getGames(years=list(range(2015,dt.datetime.now().year +1,1)),link="https://gol.gg/tournament/tournament-matchlist/LPL%20Spring%20{}/"):
  df = pd.DataFrame()
  for year in years:
    result = pd.read_html(link.replace('{}',str(year)))
    df = df.append(result,ignore_index=True)
  return df
# %%
games = getGames()
df = set_all_features(games)
teams_df = get_teams_df(df)
# %%
