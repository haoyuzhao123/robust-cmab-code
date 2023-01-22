import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('exp_name', type=str) ## CascadeBandit/SetCover/SpaningTree/ShortestPath
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

# plot random exp
exp_num = 0
while os.path.exists(os.path.join('./SimulationResults', args.exp_name, args.exp_type + str(exp_num) + '.csv')):
    data = pd.read_csv(os.path.join('./SimulationResults', args.exp_name, args.exp_type + str(exp_num) + '.csv'))
    exp.append(data)
    exp_num += 1


df = pd.concat(exp, axis=0, ignore_index=True)
print(df.shape)

grouped_df_mean = df.groupby(["Time(Iteration)"]).mean()
grouped_df_std = df.groupby(["Time(Iteration)"]).std()
quant_num = 0.2
grouped_df_quantile_min = df.groupby(["Time(Iteration)"]).quantile(quant_num)
grouped_df_quantile_max = df.groupby(["Time(Iteration)"]).quantile(1-quant_num)

colors = list(mcolors.TABLEAU_COLORS.keys())
cols = list(grouped_df_mean.columns)

for c in range(len(cols)):
    plt.plot(range(grouped_df_mean.shape[0]), grouped_df_mean[cols[c]], label='Random Target', color=colors[c])
    # plt.fill_between(grouped_df_std.index, grouped_df_mean[cols[c]] - grouped_df_std[cols[c]], grouped_df_mean[cols[c]] + grouped_df_std[cols[c]], color=colors[c], alpha=0.2)
    plt.fill_between(grouped_df_std.index, grouped_df_quantile_min[cols[c]], grouped_df_quantile_max[cols[c]], color=colors[c], alpha=0.2)

#plot another exp
target = "second"
exp_num = 0
if os.path.exists(os.path.join('./SimulationResults', args.exp_name, args.exp_type + target + str(exp_num) + '.csv')):
    exp = []
    while os.path.exists(os.path.join('./SimulationResults', args.exp_name, args.exp_type + target + str(exp_num) + '.csv')):
        data = pd.read_csv(os.path.join('./SimulationResults', args.exp_name, args.exp_type + target + str(exp_num) + '.csv'))
        exp.append(data)
        exp_num += 1
    print(exp_num)

    df = pd.concat(exp, axis=0, ignore_index=True)
    print(df.shape)

    grouped_df_mean = df.groupby(["Time(Iteration)"]).mean()
    grouped_df_std = df.groupby(["Time(Iteration)"]).std()
    quant_num = 0.2
    grouped_df_quantile_min = df.groupby(["Time(Iteration)"]).quantile(quant_num)
    grouped_df_quantile_max = df.groupby(["Time(Iteration)"]).quantile(1-quant_num)
    # import ipdb;ipdb.set_trace()

    colors = list(mcolors.TABLEAU_COLORS.keys())
    cols = list(grouped_df_mean.columns)

    for c in range(len(cols)):
        plt.plot(range(grouped_df_mean.shape[0])[:3000], grouped_df_mean[cols[c]][:3000], label='Second Best ST Target', color=colors[1])
        # plt.fill_between(grouped_df_std.index, grouped_df_mean[cols[c]] - grouped_df_std[cols[c]], grouped_df_mean[cols[c]] + grouped_df_std[cols[c]], color=colors[1], alpha=0.2)
        plt.fill_between(grouped_df_std.index[:3000], grouped_df_quantile_min[cols[c]][:3000], grouped_df_quantile_max[cols[c]][:3000], color=colors[1], alpha=0.2)

target = "spchosen"
exp_num = 0
if os.path.exists(os.path.join('./SimulationResults', args.exp_name, args.exp_type + target + str(exp_num) + '.csv')):
    exp = []
    while os.path.exists(os.path.join('./SimulationResults', args.exp_name, args.exp_type + target + str(exp_num) + '.csv')):
        data = pd.read_csv(os.path.join('./SimulationResults', args.exp_name, args.exp_type + target + str(exp_num) + '.csv'))
        exp.append(data)
        exp_num += 1

    df = pd.concat(exp, axis=0, ignore_index=True)
    print(df.shape)

    grouped_df_mean = df.groupby(["Time(Iteration)"]).mean()
    grouped_df_std = df.groupby(["Time(Iteration)"]).std()
    quant_num = 0.2
    grouped_df_quantile_min = df.groupby(["Time(Iteration)"]).quantile(quant_num)
    grouped_df_quantile_max = df.groupby(["Time(Iteration)"]).quantile(1-quant_num)

    colors = list(mcolors.TABLEAU_COLORS.keys())
    cols = list(grouped_df_mean.columns)

    for c in range(len(cols)):
        plt.plot(range(grouped_df_mean.shape[0]), grouped_df_mean[cols[c]], label='Specially Chosen Target', color=colors[2])
        # plt.fill_between(grouped_df_std.index, grouped_df_mean[cols[c]] - grouped_df_std[cols[c]], grouped_df_mean[cols[c]] + grouped_df_std[cols[c]], color=colors[2], alpha=0.2)
        plt.fill_between(grouped_df_std.index, grouped_df_quantile_min[cols[c]], grouped_df_quantile_max[cols[c]], color=colors[2], alpha=0.2)


print("plotting")
plt.xlabel('Iterations')
plt.ylabel(y_label)
plt.legend(loc="upper left")
plt.title(title)

print("saving")
plt.savefig(os.path.join('./SimulationResults', args.exp_name, args.exp_type + '.pdf'))
# plt.show()