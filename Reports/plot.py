import numpy as np
import matplotlib.pylab as plt

dataset = "cross_task"
trainer_type = "baas_chaos"
thres_type = "entropy"

filename = "result_mof_"+dataset+"_"+trainer_type+"_"+thres_type+".npy"
x_mof = np.load(filename)

filename = "result_mof-bg_"+dataset+"_"+trainer_type+"_"+thres_type+".npy"
x_mof_bg = np.load(filename)

plt.subplot(1,2,1)
plt.boxplot(x_mof)
plt.xticks(np.arange(1,13), ['0.0','0.5','1','1.5','2','2.5','3','3.5','4','4.5','5','5.5'])
plt.xlabel("Entropy Threshold")
# plt.xticks(np.arange(1,10), ['0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9'])
# plt.xlabel("Logit Threshold")
plt.ylabel("Accuracy %")
plt.title(trainer_type+" Result MoF")
plt.grid(b=True, which='major')

plt.subplot(1,2,2)
plt.boxplot(x_mof_bg)
plt.xticks(np.arange(1,13), ['0.0','0.5','1','1.5','2','2.5','3','3.5','4','4.5','5','5.5'])
plt.xlabel("Entropy Threshold")
# plt.xticks(np.arange(1,10), ['0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9'])
# plt.xlabel("Logit Threshold")
plt.ylabel("Accuracy %")
plt.title(trainer_type+" Result MoF-Background")
plt.grid(b=True, which='major')


plt.show()