import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

dataset = "cross_task"
trainer_type = "no_background"
thres_type = "logit"
runs = 5

x_mof = np.zeros((0,20))
x_mof_bg = np.array((0,20))
auc = np.array((0,1))
for i in range(runs):
    filename = "result_mof_"+dataset+"_"+trainer_type+"_"+thres_type+"_"+str(i)+".npy"
    x_mof = np.concatenate((x_mof, np.load(filename)), axis=0)

    filename = "result_mof-bg_"+dataset+"_"+trainer_type+"_"+thres_type+"_"+str(i)+".npy"
    x_mof_bg = np.concatenate((x_mof_bg, np.load(filename)), axis=0)

    filename = "result_auc_"+dataset+"_"+trainer_type+"_"+thres_type+"_"+str(i)+".npy"
    auc = np.concatenate((auc, np.load(filename)), axis=0)

auc = auc*100

x_mof_bg = pd.DataFrame(x_mof_bg.T, index=range(np.shape(x_mof_bg)[1]))
x_mof_bg_mean = x_mof_bg.T.mean()
if runs>1:
    x_mof_bg_std = x_mof_bg.T.std()

x_mof = pd.DataFrame(x_mof.T, index=range(np.shape(x_mof)[1]))
x_mof_mean = x_mof.T.mean()
if runs>1:
    x_mof_std = x_mof.T.std()

auc = pd.DataFrame(auc.T, index=range(np.shape(auc)[1]))
auc_mean = auc.T.mean()
if runs>1:
    auc_std = auc.T.std()

ax = x_mof_bg_mean.plot(color="red")
if runs>1:
    x_mof_bg_mean.plot(yerr=x_mof_bg_std, capsize=3, color="red")

x_mof_mean.plot(color="green")
if runs>1:
    x_mof_mean.plot(yerr=x_mof_std, capsize=3, color="green")

auc_mean.plot(color="blue")
if runs>1:
    auc_mean.plot(yerr=auc_std, capsize=3, color="blue")

ax.grid(b=True, which='major')
ax.set_xticks([n for n in range(20)])
ax.set_xticklabels([str(n*5) for n in range(20)])
if thres_type == "entropy":
    ax.set_xlabel("Entropy threshold as % of Log(# of fg classes)")
elif thres_type == "logit":
    ax.set_xlabel("Logit threshold as % of sigmoid outcome [0,1]")
ax.set_ylabel("Accuracy %")

red_patch = mpatches.Patch(color='red', label='MoF-BG')
green_patch = mpatches.Patch(color='green', label='MoF')
blue_patch = mpatches.Patch(color='blue', label='AUC %')
plt.legend(handles=[red_patch, green_patch, blue_patch])

plt.show()