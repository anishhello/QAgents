# SnakeAgent
The snake agent is a Rl model based on the Q learning algorithm which learns to play the famous snake game.

The standard baseline Q learning algorithm performed decently in the environment.

## Some changes made:

Changed the agent model from a single network to two networks namely the **target and the policy network**.The target network is basically the copy of the policy network . The only difference is that the weights of the target network are copied from the policy network after a certain number of epochs. We tried with full copying and soft copying but the results were pretty same for soft copy coefficient>0.5. This provides for a more stable training and convergence of the agent for which the metric is given by the difference in the output Q-value and target Q-value from the **Bell equation**.

Updated the standard **reward function** with special scenarios where the agent seem to stuck like falling into a endless loop and also reinforced the greedy movement towards food.Imposed additional penalities if loops are noticed.

Finetuned the diference in rates of updates of the target and policy networks.Finetuned the **epsilon greedy startegy by using an exponential function** to always have a non zero probability of exploration.

## Results 

The average score in the training process converges to nearly 30 in the setup provided.

The highest scores have crossed 60+ scores at timed during training.

![Screenshot 2024-06-15 182327](https://github.com/anishhello/QAgents/assets/133523672/28fee4e2-ec25-4392-a007-3cab733e73c2)


## Further improvements

The training as observed has alternate spikes and dips which hint at some overfitting issues.These can be mitigated by furher finetuning the reward function or the action-environment state dimensionalities.

Increase the dimensions of state and action variables along with deepening of the model used for the networks to make the overall setup less discontinuous and further optimzing the target and policy networks for optimal decisions.







# Trading Agent
