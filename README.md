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

## Part one (predicting stock prices)

* Given we know that stock prices mantain a random walk process in movement, We first make the time series stationary by differencing methods verifying with ADF tests.

* ARMA , ACF, PACF plots were used for the same.

* Created few more features like percentage change with lag 1 and moving averages for 7,14,21,30 days.

* Now tried to predict the pct-change data. Given the results with linear regression and Xgboost

![Screenshot 2024-07-10 133552](https://github.com/anishhello/QAgents/assets/133523672/aa4ae9b2-1146-4d7f-b797-70f5912ee147)
### Xgboost


![Screenshot 2024-07-10 133625](https://github.com/anishhello/QAgents/assets/133523672/6995685a-75be-47ca-9852-c6fa89961910)
### LinearRegression


* Now we created an ensemble of the linearregression and xgboost to create a balance between  the bias and variance for the model

![Screenshot 2024-07-10 133650](https://github.com/anishhello/QAgents/assets/133523672/bf2774fc-01fc-4bb9-9122-3c17fcec0073)
### Final predictions on the original price(It nearly overlaps!!!!!)


## Part two ( training an agent on this predicted data)

* For this part we chose an Rl agent based on Q learning to optimize the decisions based on the data predicted

* Again used two separate networks for policy and target networks for stable training and convergence.

* Finetuned the reward functions based on the conditions.If the prices have an upward trend overall we impose less penalty on going negative on the cumulative sum hoping for a coverup when the prices rise , basically letting the agent "breathe" . In the other case when overall trend is constant or lower we impose high penalty on going negative beacuse it will prevent unnecesaary trials where agent takes buy positions initially to suffer heavy loss afterwards. This is also done to cap the initial pocket to initiate the trade.

![Screenshot 2024-07-10 132526](https://github.com/anishhello/QAgents/assets/133523672/7ca81e44-2252-4cf9-937d-bcd7eb0b1af4)
### exemplar results during training

## Further Improvements

* extending the dimensions for state varibales by considering informations from advanced signals like MACD,RSI etc.

* extending the action variables buy making a portfoliio of stocks rather than using a single one to counter risky positions in individual stocks.

 

  



