import pandas as pd
import datetime
import streamlit as st
import numpy as np

def configure_df(games):
  df = games.copy()
  # Date
  df.Date = pd.to_datetime(df['Date'])
  df = df.sort_values(by='Date')
  df = df[df.Score != '-']
  df = df[df.Date < (datetime.datetime.now() - datetime.timedelta(days=1))]
  #Rename column
  df.rename(columns={'Blue Side':'blue','Red Side':'red','Date':'date'},inplace=True,errors='ignore')
  # Spliting scores and set winner
  df[['blue_score','red_score']] = df.Score.str.split(' - ',expand=True)
  df[['blue_score','red_score']] = df[['blue_score','red_score']].astype(int)
  df['winner'] = df.apply(lambda x : 'blue' if x.blue_score > x.red_score else 'red',axis=1 )
  df['winner_name'] = df.apply(lambda x : x.blue if x.blue_score > x.red_score else x.red,axis=1 ) 
  df['blue_is_winner'] = df['winner'] == 'blue'
  df['year'] = df.date.dt.year
  #Unecessary columns
  df.drop(columns=['Unnamed: 4','Patch','Game','Score'],inplace=True,errors='ignore')
  #Set types category
  df[['blue','red','winner','winner_name']] = df[['blue','red','winner','winner_name']].astype('category')
  return df

def get_past_team_stats(row,old_df):
    # Get only past information
    past_df = old_df[old_df.date < row.date ]

    blue_plays = past_df[(past_df.blue ==row.blue) | (past_df.red ==row.blue) ]
    red_plays = past_df[(past_df.blue ==row.red) | (past_df.red ==row.red) ]

    # Wins
    blue_wins =  len(past_df[(past_df.winner_name ==row.blue)])
    red_wins =  len(past_df[(past_df.winner_name ==row.red)])

    blue_win_avg = blue_wins/len(blue_plays) if len(blue_plays) != 0 else 0
    red_win_avg = red_wins/len(red_plays) if len(red_plays) != 0 else 0

    blue_won_last_match = False if len(blue_plays) == 0 else blue_plays.iloc[-1].winner_name == row.blue
    red_won_last_match = False if  len(red_plays) == 0 else red_plays.iloc[-1].winner_name == row.red 

    return blue_win_avg,red_win_avg,blue_won_last_match,red_won_last_match # blue_wins,red_wins,

def set_past_team_stats(old_df):
    df = old_df.copy()
    df[['blue_win_avg','red_win_avg','blue_won_last_match','red_won_last_match']] = df.apply(lambda x: get_past_team_stats(x,df),axis=1,result_type='expand')
    return df


def set_teams_streak(old_df):
    new_df = old_df.copy()
    team_df = pd.DataFrame(set([*new_df['blue'].values,*new_df['red'].values]),columns=['team'])
    def count_team_streak(team,df):
        plays = df.copy()
        plays = plays[(plays.blue == team) | (plays.red == team)][['winner_name']]

        plays['start_of_streak'] = plays.winner_name.ne(plays['winner_name'].shift(-1))
        plays['streak_id'] = plays['start_of_streak'].cumsum()
        plays['streak_counter'] = plays.groupby('streak_id').cumcount() + 1
        return plays['streak_counter']

    for team in team_df.itertuples():
        new_df.loc[new_df.winner_name == team.team,'streak_counter'] = count_team_streak(team.team,new_df)

    new_df['blue_streak'] = new_df.apply (lambda x: x.streak_counter if x.winner_name == x.blue else 0,axis=1)
    new_df['red_streak'] = new_df.apply (lambda x: x.streak_counter if x.winner_name == x.red else 0,axis=1)

    for i, row in new_df.iterrows():
        if row.blue_streak > 0:
            new_df.at[i,'blue_streak'] = (row.blue_streak - 1)
        if row.red_streak > 0:
            new_df.at[i,'red_streak'] = (row.red_streak - 1)
    return new_df.drop(columns=['streak_counter'])

def get_past_wins_against(row,side,old_df):
  df = old_df.copy()
  past_df = df[df.date < row.date]
  other = 'red' if side == 'blue' else 'blue'
  wins = len(past_df[(past_df.winner_name ==row[side]) & ( (past_df.red ==row[other]) | (past_df.blue ==row[other])) ])
  return wins

def set_past_wins_against(old_df):
  df = old_df.copy()
  df['blue_winst_against'] = df.apply(lambda x : get_past_wins_against(x,'blue',df),axis=1) 
  df['red_winst_against'] = df.apply(lambda x : get_past_wins_against(x,'red',df),axis=1)
  return df

# Get metrics of each team of lol
def get_teams_df(df):
  team_df = pd.DataFrame(set([*df['blue'].values,*df['red'].values]),columns=['team'])
  for team in team_df.itertuples():
    matches = df[(df.blue == team.team) | (df.red == team.team) ]
    plays = len(matches)
    wins = len(matches[matches.winner_name == team.team])
    losses = len(matches[matches.winner_name != team.team])

    team_df.loc[team_df.team == team.team,'plays' ] =plays
    team_df.loc[team_df.team == team.team,'wins' ] = wins
    team_df.loc[team_df.team == team.team,'losses' ] = losses

    team_df.loc[team_df.team == team.team,'win_avg' ] = team_df.wins / team_df.plays 
    team_df = team_df.sort_values(by='plays',ascending=False)
    
  return team_df


def get_last_team_game(team,df):
  last_game = df[(df.blue == team) | (df.red == team)].iloc[-1]
  oponent = 'blue' if last_game.red == team else 'red'
  side = 'blue' if oponent == 'red' else 'red'
  side_columns = last_game[last_game.index.str.contains(side,case=False)].index
  last_game =  last_game[[*side_columns]]
  last_game = last_game.rename(lambda x: x.lower().replace(f'{side}_',''))
  return last_game


def rolling_win_avg(row,df,window,team):
    past_df = df[df.date < row.date ]
    side_plays = past_df[(past_df.blue == team) | ( past_df.red == team)]
    last_n_plays = side_plays[window*-1:]
    wins = len(last_n_plays[(last_n_plays.winner_name == team)])
    wins_avg = wins / len(last_n_plays) if len(last_n_plays) != 0 else 0
    return wins_avg

def set_rolling_wins_avg(old_df,windows=[5,10]):
    df = old_df.copy()

    for window in windows:
        df[f'blue_rolling_win_avg_{window}'] =  df.apply(lambda x: rolling_win_avg(x,df,window,x.blue),axis=1 )
        df[f'red_rolling_win_avg_{window}'] =  df.apply(lambda x: rolling_win_avg(x,df,window,x.red),axis=1 )
        df[f'blue_rolling_win_avg_{window}'] = df[f'blue_rolling_win_avg_{window}'].fillna(0)
        df[f'red_rolling_win_avg_{window}'] = df[f'red_rolling_win_avg_{window}'].fillna(0)
    return df

@st.cache
def set_all_features(games):
    df = configure_df(games)
    df = set_past_team_stats(df)
    df = set_teams_streak(df)
    df = set_past_wins_against(df)
    df = set_rolling_wins_avg(df)
    return df


