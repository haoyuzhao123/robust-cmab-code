import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import argparse

# SMALL_SIZE = 18
# MEDIUM_SIZE = 15
BIGGER_SIZE = 15

plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('legend', fontsize=20)    # legend fontsize
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels


# plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
parser = argparse.ArgumentParser()

exp = []

# plot random exp
data = pd.read_csv(os.path.join('./SimulationResults/ShortestPathAttackability/attackability_walk.csv'))
data = data.iloc[:, 1:]
num_exp = 20
percent = []
keys = [25, 50, 75, 100]
for c in keys:
    p = 0
    for i in data[str(c)]:
        p = p + 1 if i == 1 else p
    percent.append(p/num_exp * 100)

print(percent)

plt.bar(range(len(keys)), percent, width=0.4)
plt.xticks(range(len(keys)), keys)
plt.xlabel('Walk Length')
plt.ylabel('Percentage')
plt.title('Percentage of Attackable Scenarios')
plt.tight_layout()
print("saving")
plt.savefig(os.path.join('./SimulationResults/ShortestPathAttackability/attackability.png'))