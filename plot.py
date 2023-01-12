import pandas as pd
import seaborn as sns
import os

exp = []
expName = 'CascadeBandit'
exp_num = 0

while os.path.exists('./SimulationResults/' + expName + str(exp_num) + '.csv'):
    data = pd.read_csv('./SimulationResults/' + expName + str(exp_num) + '.csv')
    exp.append(data)
    exp_num += 1


df = pd.concat(exp, axis=0, ignore_index=True)
print(df)

rplot = sns.lineplot(data=df, x="Time(Iteration)", y="CascadeUCB-V", label='CascadeUCB-V', color='blue')
rplot = sns.lineplot(data=df, x="Time(Iteration)", y="CascadeUCB-V-Attack", label='CascadeUCB-V-Attack', color='red')

rplot.set_xlabel('Iterations')
rplot.set_ylabel('Regret')

fig = rplot.get_figure()
fig.savefig("./SimulationResults/out.png") 