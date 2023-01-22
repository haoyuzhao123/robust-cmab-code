import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import argparse

plt.rcParams.update({'font.size': 12})

parser = argparse.ArgumentParser()
parser.add_argument('exp_name', type=str) ## CascadeBandit/SetCover/
parser.add_argument('exp_type', type=str) ## Reward/Regret/Cost/Rate

args = parser.parse_args()

exp = []

if args.exp_type == 'Regret':
    y_label = args.exp_type
    title = "Cumulative Regret"

if args.exp_type == 'Reward':
    y_label = args.exp_type
    title = "Average Reward"

if args.exp_type == 'Cost':
    y_label = args.exp_type
    title = "Total Cost"

if args.exp_type == 'Rate':
    y_label = 'Count'
    title = "Number of times Target Arm is Played"

exp_num = 0

while os.path.exists(os.path.join('./SimulationResults', args.exp_name, args.exp_type + str(exp_num) + '.csv')):
    data = pd.read_csv(os.path.join('./SimulationResults', args.exp_name, args.exp_type + str(exp_num) + '.csv'))
    exp.append(data)
    exp_num += 1

df = pd.concat(exp, axis=0, ignore_index=True)[::-1]

cols = df.columns.tolist()
cols = cols[::-1]
df = df[cols]

print(df)

grouped_df_mean = df.groupby(["Time(Iteration)"]).mean()
grouped_df_std = df.groupby(["Time(Iteration)"]).std()

quant_num = 0.3
grouped_df_quantile_min = df.groupby(["Time(Iteration)"]).quantile(quant_num)
grouped_df_quantile_max = df.groupby(["Time(Iteration)"]).quantile(1-quant_num)


colors = list(mcolors.TABLEAU_COLORS.keys())
cols = list(grouped_df_mean.columns)

for c in range(len(cols)):
    plt.plot(range(grouped_df_mean.shape[0]), grouped_df_mean[cols[c]], label=cols[c], color=colors[c])
    # plt.fill_between(grouped_df_std.index, grouped_df_quantile_max[cols[c]], grouped_df_quantile_min[cols[c]], color=colors[c], alpha=0.2)

    plt.fill_between(grouped_df_std.index, (grouped_df_mean[cols[c]] - grouped_df_std[cols[c]]).clip(0, None), grouped_df_mean[cols[c]] + grouped_df_std[cols[c]], color=colors[c], alpha=0.2)

    

print("plotting")
plt.xlabel('Iterations')
plt.ylabel(y_label)
plt.legend(loc="upper left")
plt.title(title)
plt.tight_layout()

print("saving")
plt.savefig(os.path.join('./SimulationResults', args.exp_name, args.exp_type + '.pdf'))
# plt.show()

