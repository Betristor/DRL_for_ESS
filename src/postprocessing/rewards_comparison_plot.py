import numpy as np
import matplotlib.pyplot as plt
from pickle import load

# DQN type list
dqn_types = ['vanilla', 'double_dueling']

# empty dictionary for rewards per model
model_rewards = {}

# load results for each DQN type
for model in dqn_types:
	rewards = open(f"/Users/yuhengzhang/Documents/博一上/Foundations of RL/Project/DRL_for_ESS/results/rewards/rewards_{model}.pkl", "rb") 
	model_rewards[model] = load(rewards)
	rewards.close()

# dictionary to store reward rolling average
model_rewards_av = {}

# moving average function
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

# get rolling average reward over each model type 
for model in dqn_types:
	model_rewards_av[model] = moving_average(model_rewards[model],50)

# colour scheme for plot 
colours = ['#f8cf01','#f08080', 'dodgerblue', 'red']

# declare figure
plt.figure(figsize=(13, 5))
# set graph style
plt.style.use(['seaborn-whitegrid'])

# intialise plot axis
ax = plt.subplot()

# plot average rewards
for idx, model in enumerate(dqn_types):
	print(model)
	ax.plot(model_rewards_av[model], label=f'{model}', color=colours[idx], linewidth=1.75, alpha=0.8)

# apply graph formatting
ax.grid(True, alpha=0.6, which="both")
ax.spines['bottom'].set_color('black')  
ax.spines['top'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['right'].set_color('black')
ax.tick_params(direction="out", length=2.0)

ax.tick_params(axis='y', labelsize= 8)
ax.tick_params(axis='x', labelsize= 8)
ax.set_ylabel('Reward', fontsize=9)
ax.set_xlabel('Episode', fontsize=9)
ax.grid(alpha=0.3)

ax.set_xlim([0, 5000])
ax.tick_params(direction="out", length=2.0)
ax.set_xticks(np.arange(0, 5001, 250))

ax.set_ylim([-2, 7])
ax.tick_params(direction="out", length=2.0)
ax.set_yticks(np.arange(-2, 7.1, 1))

handle1, label1 = ax.get_legend_handles_labels()

leg = ax.legend(handle1, labels=["DQN", "Duel DQN"], loc="lower right", fontsize=7, frameon=True)
leg.set_zorder(5)

frame = leg.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('white')

plt.title("DQN v.s. Duel DQN", fontsize=10)
plt.savefig('rewards_comparison_plot.png', bbox_inches='tight', transparent=True)
plt.show()


