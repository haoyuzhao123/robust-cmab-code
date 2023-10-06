import time
import os
import pickle 
import datetime
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from IMconf import *
from Tool.utilFunc import *
import random

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import argparse

data = pd.read_csv('./SimulationResults/IM_seed_size_5/Cost0.csv')
cols = list(data.columns)

print(cols)

for c in range(1, len(cols)):
    plt.plot(data[cols[0]], data[cols[c]], label=cols[c])

print("plotting")
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.legend(loc="upper left")
plt.title('Total Cost')

plt.savefig('./SimulationResults/IM_seed_size_5/cost.png')