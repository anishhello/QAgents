# SnakeAgent
The snake agent is a Rl model based on the Q learning algorithm which learns to play the famous snake game.

The standard baseline Q learning algorithm performed decently in the environment.

## Some changes made:

Changed the agent model from a single network to two networks namely the **target and the policy network**.The target network is basically the copy of the policy network . The only difference is that the weights of the target network are copied from the policy network after a certain number of epochs. We tried with full copying and soft copying but the results were pretty same for soft copy coefficient>0.5. This provides for a more stable training and convergence of the agent for which the metric is given by the difference in the output Q-value and target Q-value from the **Bell equation**.

Updated the standard **reward function** with special scenarios where the agent seem to stuck like falling into a endless loop and also reinforced the greedy movement towards food.






# Trading Agent
