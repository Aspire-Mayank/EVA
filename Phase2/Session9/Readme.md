# ***Lets look at these 15 steps through code of Twin Delays Deep Deterministic Algorithms:***

## STEP 1 : We initialize the Experience Replay Memory with a size of 1e6. Then we populate it with new transitions with all tuples as input (s', a, r, s) as tensor.
![alt text](https://github.com/Aspire-Mayank/EVA/blob/master/Phase2/Session9/step1.PNG?raw=true)

## STEP 2 : Build one DNN for the Actor Model and one for Actor Target
![alt text](https://github.com/Aspire-Mayank/EVA/blob/master/Phase2/Session9/step2.PNG?raw=true)

## STEP 3 : Build two DNNs for the two Critic model and two DNNs for the two critic Targets
![alt text](https://github.com/Aspire-Mayank/EVA/blob/master/Phase2/Session9/step3.PNG?raw=true)

## STEP 4 -15 : Training process. Create a T3D class, initialize variables and get ready for step 4

## STEP 4 : sample from a batch of transition(s,s',a, r) from the memory
![alt text](https://github.com/Aspire-Mayank/EVA/blob/master/Phase2/Session9/step4-15.PNG?raw=true)

## STEP 5 : From the next state s', the actor target plays the next action a'
![alt text](https://github.com/Aspire-Mayank/EVA/blob/master/Phase2/Session9/step4.PNG?raw=true)

## Step 6 : We add Gaussian noise to this next action a' and we clamp it in a range of values supported by the environment
![alt text](https://github.com/Aspire-Mayank/EVA/blob/master/Phase2/Session9/step5.PNG?raw=true)

## STEP 7 : The Two Critic targets take each the couple (s', a') as input and return two Q values, Qt1(s', a') and Qt2(s', a') as outputs.
![alt text](https://github.com/Aspire-Mayank/EVA/blob/master/Phase2/Session9/step6.PNG?raw=true)
> ***This is not target_Q, we are just being lazy, and want to use the same variable name later on.***

## STEP 8 : Keep the minimum of these two Q-Values
![alt text](https://github.com/Aspire-Mayank/EVA/blob/master/Phase2/Session9/step7.PNG?raw=true)

## STEP 9 : We get the final target of the two Critic models, which is 
![alt text](https://github.com/Aspire-Mayank/EVA/blob/master/Phase2/Session9/step8.PNG?raw=true)
>  ***Target Qt = Reward + (gamma * min(Q1, Q2))***

> ***A:-> we can define "target_q" as "Qt" as "reward + discount * torch.min(Q1, Q2)" but it won't work.*** 
> ***1. First, we are only supposed to run this if the episode is over, which means we need to intergate Done.***

> ***2. Second, target_q would create it's BP/computation graph, and without detaching Qt1/Qt2 from their own graph, we are complicating things, i.e we need to use detach. 
>   let's look below:***

## STEP 10 : Two critic models each take the couple(s, a) as input and return two Q-values
![alt text](https://github.com/Aspire-Mayank/EVA/blob/master/Phase2/Session9/step9.PNG?raw=true)

## STEP 11 : Compute the critic loss
***we compute the loss coming from the two critic models***
![alt text](https://github.com/Aspire-Mayank/EVA/blob/master/Phase2/Session9/step10.PNG?raw=true)

## STEP 12 : Backpropagate this critic loss and update the parameters of two critic models with Adam optimizer
![alt_text](https://github.com/Aspire-Mayank/EVA/blob/master/Phase2/Session9/step11.PNG?raw=ture)

## STEP 13 : once every two iterations, we update our Actor model by performing gradient Ascent on the output of the first Critic model.
![alt_text](https://github.com/Aspire-Mayank/EVA/blob/master/Phase2/Session9/step12.PNG?raw=true)

## Step 14 : Still once every two iterations, we update the weights of the Actor target by Polyak Averaging
![alt_text](https://github.com/Aspire-Mayank/EVA/blob/master/Phase2/Session9/step13.PNG?raw=true)

## Step 15 : Still once every two iterations, we update the weights of the Critic target by Polyak Averaging
![alt_text](https://github.com/Aspire-Mayank/EVA/blob/master/Phase2/Session9/step14.PNG?raw=true)

# T3D Twin Delayed Deterministic Deep Policy Gradient or DDPG Done Here !..



## Keep Learning Keep Sharing.. All the best. :) Happy Learning...
