import pandas as pd 
def add_to_csv(dir, index, value):
    df = pd.read_csv(dir, index_col=False)
    # print(df.add(['Oxford', 'MPC', 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))
    # print(df.append(['Oxford', 'MPC', 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))
    df.loc[len(df)] = ('Oxford', 'MPC', 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    # print(df)
    df.to_csv(dir, index=False)
    # df[index] = value
    # (df.T).to_csv(dir, index=True)
dir = 'experiment/results_2023-02-17 01:15:56.116348.csv'
index = ('Oxford', 'MPC', 1, 1, 1, 1)
value = [1, 1, 1, 1, 1, 1, 1]
add_to_csv(dir, index, value)