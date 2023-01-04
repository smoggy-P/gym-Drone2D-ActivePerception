import pandas as pd
import csv

df = pd.read_csv('./experiment/results.csv', index_col=['Method','Map','Number of agents']).T



df[('LookAhead',1,10)] = {'Success': 0, 'Static Collision': 0, 'Dynamic Collision': 0}
# df.to_csv("./experiment/results.csv", index=False)

# method = "LookAhead"
# map_idx = 1
# agent_num = 10
(df.T).to_csv("./experiment/results.csv", index=False)
# for index, row in df.iterrows():
#     print("index:", index)



# reader = csv.DictReader(open('./experiment/results.csv'))
# for row in reader:
#     print(row)