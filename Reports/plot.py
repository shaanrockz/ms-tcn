import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

dataset = "cross_task"
trainer_type = "baas_chaos"
thres_type = "logit"
runs = 1

filename = "result_mof_"+dataset+"_"+trainer_type+"_"+thres_type+".npy"
x_mof = np.load(filename)

filename = "result_mof-bg_"+dataset+"_"+trainer_type+"_"+thres_type+".npy"
x_mof_bg = np.load(filename)


x_mof_bg = pd.DataFrame(x_mof_bg.T, index=range(np.shape(x_mof_bg)[1]))
x_mof_bg_mean = x_mof_bg.T.mean()
if runs>1:
    x_mof_bg_std = x_mof_bg.T.std()

x_mof = pd.DataFrame(x_mof.T, index=range(np.shape(x_mof)[1]))
x_mof_mean = x_mof.T.mean()
if runs>1:
    x_mof_std = x_mof.T.std()

ax = x_mof_bg_mean.plot(color="red")
if runs>1:
    x_mof_bg_mean.plot(yerr=x_mof_bg_std, capsize=3, color="red")

x_mof_mean.plot(color="green")
if runs>1:
    x_mof_mean.plot(yerr=x_mof_std, capsize=3, color="green")

ax.grid(b=True, which='major')
ax.set_xticks([n for n in range(20)])
ax.set_xticklabels([str(n*5) for n in range(20)])
# ax.set_xlabel("Entropy threshold as % of Log(# of fg classes)")
ax.set_xlabel("Logit threshold as % of sigmoid outcome [0,1]")
ax.set_ylabel("Accuracy %")

red_patch = mpatches.Patch(color='red', label='MoF-BG')
green_patch = mpatches.Patch(color='green', label='MoF')
plt.legend(handles=[red_patch, green_patch])

plt.show()