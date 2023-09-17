import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import ast
import seaborn as sns
from itertools import product

directory_path = "./experiment/validation/"
csv_files = [file for file in os.listdir(directory_path) if file.endswith(".csv")]
dataframes = []
for file in csv_files:
    file_path = os.path.join(directory_path, file)
    dataframe = pd.read_csv(file_path)
    dataframes.append(dataframe)
df = pd.concat(dataframes, ignore_index=True)
df['Planner'] = df['Planner'].replace({'Primitive' : 'Global Primitive [7]', 'Jerk_Primitive': 'Local Primitive [10]', 'MPC': 'MPC [11]'})
df['Method'] = df['Method'].replace({'NoControl': 'FullRange', 'LookAhead':'LookAhead [16]', 'LookGoal':'LookGoal [18]', 'Oxford':'Finean et al. [3]', 'Owl':'Owl [10]'})
df['Method_Planner'] = df['Method'] + "+" + df['Planner']
df = df[(df['Motion Profile'] == 'CVM')]
unique_method_planner = df['Method_Planner'].unique()
linestyles = ['-', '--', '-.', ':', '-', '--', '-', ':', '-', '--', '-', ':', '-']
markers = ['o', 's', '^', 'v', '<', '>', 'p', '*', 'h', 'H', 'D', 'd', '+']
colors = sns.color_palette('hsv', len(unique_method_planner))
color_map = dict(zip(unique_method_planner, colors))

sort_columns = ['Map ID', 'Number of agents', 'Agent size', 'Agent speed', 'Motion Profile']
plt.rc('font', family='serif')
plt.rc('text', usetex=True)
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16

metric_dir_dict = {'Survivability(reversed)': 'experiment/metrics/survivability/metrics_6m_12s_CVM.csv', 
                   'Traversibility(reversed)': 'experiment/metrics/traversibility/traversibility.csv',
                   'VO Feasibility(reversed)': 'experiment/metrics/vo/vo.csv',
                   'Obstacle Density': 'experiment/metrics/density.csv',
                   'Dynamic Traversability(reversed)':'experiment/metrics/traversibility/traversibility_multiple.csv'}

fig, subfigs = plt.subplots(nrows=2, ncols=3, figsize=(15, 8), )

for j, metric in enumerate(['Obstacle Density', 'Traversibility(reversed)', 'Dynamic Traversability(reversed)', 'VO Feasibility(reversed)', 'Survivability(reversed)', 'Global Survivability(reversed)']):

    metric_dict = {}
    round_metrics_dict = {}

    if metric_dir_dict.get(metric, None) is None:
        a = np.load("experiment/metrics/survivability/collision_states_6m_12s_RVO.npy")
        index = 0
        for map_id in range(20):
            for (agent_num, agent_size, agent_vel) in product([10, 20, 30], [5, 10, 15], [20, 40, 60]):
                survive_times = []
                for start_time in range(0, 240, 40):
                    survive_time = 0
                    while (start_time + survive_time < 240) and (not (a[index,::2,::2,start_time+survive_time].any())):
                        survive_time += 1
                    survive_times.append(survive_time)
                metric_dict[(map_id, agent_num, agent_size, agent_vel, 'CVM')] = np.mean(survive_times)
                round_metrics_dict[(map_id, agent_num, agent_size, agent_vel, 'CVM')] = round(np.mean(survive_times))
                index += 1
    else:
        df_metric = pd.read_csv(metric_dir_dict[metric])
        metrics = []
        for i in range(20):
            metrics.append(ast.literal_eval(df_metric['metric'][i]))
        data = np.array(metrics).flatten()
        index = 0
        for map_id in range(20):
            for (agent_num, agent_size, agent_vel) in product([10, 20, 30], [5, 10, 15], [20, 40, 60]):
                metric_dict[(map_id, agent_num, agent_size, agent_vel, 'CVM')] = data[index]
                index += 1


    max_value = max(metric_dict.values())
    min_value = min(metric_dict.values())
    if metric == 'Obstacle Density':
        for key in metric_dict.keys():
            metric_dict[key] = 10 - 10 * (metric_dict[key] - min_value) / (max_value - min_value)
            round_metrics_dict[key] = round(metric_dict[key])
    else:
        for key in metric_dict.keys():
            metric_dict[key] = 10 * (metric_dict[key] - min_value) / (max_value - min_value)
            round_metrics_dict[key] = round(metric_dict[key])

    df['round difficulty'] = [round_metrics_dict.get(tuple(x), 'N/A') for x in df[sort_columns].values]
    df['map difficulty'] = [metric_dict.get(tuple(x), 'N/A') for x in df[sort_columns].values]

    grouped = df.groupby(['Method_Planner', 'round difficulty', *sort_columns])
    df_grouped = grouped.agg({'Success': 'mean'}).reset_index()
    grouped = df_grouped.groupby(['Method_Planner', 'round difficulty'])['Success']

    # Create a DataFrame for results
    df_results = pd.DataFrame({
        'Mean Success': grouped.mean(),
        'Variance Success': 0 if grouped.var() is None else grouped.var()
    }).reset_index()
    df_results = df_results.fillna(0)

    cvs = []
    for i, method_planner in enumerate(unique_method_planner):
        df_plot = df_results[df_results['Method_Planner'] == method_planner]
        subfigs[j//3,j%3].plot(10 - df_plot['round difficulty'].to_numpy(), df_plot['Mean Success'].to_numpy(), 
                    color=color_map[method_planner], label=method_planner,
                    linestyle=linestyles[i],
                    marker=markers[i],
                    markersize=8,
                    linewidth=2)
        
        # Here we create the shaded variance region
        subfigs[j//3,j%3].fill_between(10 - df_plot['round difficulty'], 
                        df_plot['Mean Success'] - np.sqrt(df_plot['Variance Success']), 
                        df_plot['Mean Success'] + np.sqrt(df_plot['Variance Success']), 
                        color=color_map[method_planner], alpha=0.2)

    subfigs[j//3,j%3].set_xlabel(metric+' Metric', fontsize=18)
    if j%3 == 0:
        subfigs[j//3,j%3].set_ylabel('Success Rate', fontsize=18)
    # subfigs[j//3,j%3].set_ylabel('Success Rate', fontsize=20)
    # subfigs[j//3,j%3].set_xticks(fontsize=20)
    # subfigs[j//3,j%3].set_yticks(fontsize=20)
    subfigs[j//3,j%3].set_ylim(0, 1)
    subfigs[j//3,j%3].set_xlim(0, 10)
    # plt.tick_params(axis='both', which='major', labelsize=16)

fig.tight_layout()
lgd = plt.legend(fontsize=13.7, bbox_to_anchor=(1, 2.7), borderaxespad=0., ncol = 4)
fig.savefig('results.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')
