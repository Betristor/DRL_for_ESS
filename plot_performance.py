from pickle import load
import matplotlib.pyplot as plt

# DQN models
dqn_models = ['vanilla', 'double_dueling']

pkl = open('/Users/yuhengzhang/Documents/博一上/Foundations of RL/Project/DRL_for_ESS/results/rewards/profits.pkl', 'rb')
dqn_model_profits = load(pkl)
pkl.close()

# plot profit comparison
for idx, model in enumerate(dqn_models):
	plt.plot(dqn_model_profits[model])

plt.xlabel('Episode')
plt.ylabel('Profit')
plt.legend(labels=["DQN", "Duel DQN"], loc="lower right")
plt.show()