import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt

variables = {}

# loop to generate multiple
exec(open('test.py').read(), variables)

mytorch_data = variables['mytorch_trainer'].summary
pytorch_data = variables['pytorch_trainer'].summary

df_my = pd.DataFrame.from_dict(mytorch_data)
df_py = pd.DataFrame.from_dict(pytorch_data)

df_errs = pd.merge(
    df_my[['Test error', 'Train error', 'Epoch']],
    df_py[['Test error', 'Train error', 'Epoch']],
    on='Epoch'
)
df_errs.columns = ['Test MyTorch', 'Train Mytorch', 'Epoch', 'Test PyTorch', 'Train PyTorch']

sns.set(font_scale=1)
sns.set_style('whitegrid')
data_acc = df_errs.melt('Epoch', var_name='variable', value_name='Error %')
ax = sns.lineplot(x='Epoch', y='Error %', hue='variable', data=data_acc)

#ax.get_figure().savefig("./report/fig/err2.pdf")

import torch
n = 10
errors_my = []
errors_py = []
for i in range(n):
    variables = {}
    exec(open('test.py').read(), variables)
    errors_my.append(variables['mytorch_trainer'].summary['Test error'][-1])
    errors_py.append(variables['pytorch_trainer'].summary['Test error'][-1])

errors_my = torch.tensor(errors_my)
errors_py = torch.tensor(errors_py)
print(errors_my.std(), errors_my.mean())
print(errors_py.std(), errors_py.mean())

