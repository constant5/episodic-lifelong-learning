import pandas as pd

for i in range(4):
    df = pd.read_csv('data/ordered_data/test/' + str(i+1) +'.csv')
    print(df)
    df.dropna(subset=['content'],inplace=True)
    print(df)
    df.to_csv('data/ordered_data/test/' + str(i+1) +'.csv',index=False)