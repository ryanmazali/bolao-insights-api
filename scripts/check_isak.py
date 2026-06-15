import pandas as pd

goals = pd.read_csv('data/raw/goalscorers.csv')
sweden = goals[goals['team'] == 'Sweden'].tail(10)
print(sweden[['date', 'scorer', 'team']].to_string())