'''



Reinforcement Learning with Gym



In this file, I'll be explaining and implementing reinforcement learning,
using OpenAI's Gym. I'll be doing this with a .py file, rather than a Jupyter
notebook, because I've been having problems getting OpenAI's Gym to work in
Jupyter.



Reinforcement Learning as a Process



Reinforcement learning is about making optimal decisions, given previous
experiences. The agent takes into account its environment, decides what to do,
recieves a reward or penalty, updates its strategy accordingly, and repeats
until it has found an optimal strategy.




Self-Driving Cab



To see reinforcement learning in action, let's build a simulation of a
self-driving cab.

Our cab's job is to pick up a passenger at one location, and drop them off at
another. Let's set some rules for our cab:
1. Try and pick up, and drop off, each passenger in as short a time as possible.
2. Obey traffic rules.


Rewards


We need to set the rewards and penalties for our agent:
1. The agent recieves a high reward for a succesful drop off. The reward is high
because this is the most desirable behaviour.
2. The agent recieves a high penalty if it drops off a passenger at the wrong
location.
3. The agent recieves a slight penalty at each time step, encouraging it to
reach its desired destination in as few steps as possible. The penalty is only
slight because this behaviour is not as important as dropping off the passenger
to the correct location. We would rather the agent were slow, and drop the
passenger off in the right location, than it trying to get the passenger out of
the cab as quickly as possible, potentially dropping them off in the wrong
location.
4. The agent recieves a slight penalty for crashing into walls.


State


Our agent needs to decide what to do, depending on its location and situation.
E.g., if the cab picks up a passenger at point A, and needs to deliver the
passenger to point B, then it should take steps that move it towards point B.
The location in this example is at which 2-dimensonal point the agent is
currently at, and the situation is whether the cab has a passenger or not.

The set of all possible states our agent can inhibit is known as the state
space.

Let's build our environment. We will be using the Taxi environment from
OpenAI's Gym library. The environement will be represented by a 5 x 5 grid,
giving us 25 locations. Each corner will be a pick-up or drop-off location.
There are a few walls blocking direct routes from one square to another, forcing
our agent to learn to go around them. There are 5 different states for our
passenger to be in: any of the 4 pick-up locations, or in the cab. Taking the
number of locations, and multiplying it by the number of possible passenger
states, we have 25 x 5 = 125 possible states. Finally, our drop-off location can
be at any of the 4 designated pick-up / drop-off locations, making our total
number of possible states: 125 x 4 = 500.


Action Space


The action space is the list of all possible actions our agent can take in any
state. There are 6 actions our agent can make at any moment:
1. move north
2. move east
3. move south
4. move west
5. pick up passenger
6. drop off passenger.



Implementation



Due to the lack of Windows support for Gym, the visuals are quite underwhelming
when rendering the enivronment. For this reason, I will instead try my best to
explain what is happening in the environment, rather than show you.

Our agent recieves +20 points for a succesful drop off, loses 1 point at each
time step, and loses 10 points for attempting to pick-up a passenger, or drop
them off, in the wrong location.

The environment is as described previously, it is a 5x5 grid with the letter R
in the top left corner, the letter G in the top right corner, the letter Y in
the bottom left corner, and the letter B in the bottom right corner. Our cab is
represented by a filled square which is coloured green when a passenger is
inside, and yellow otherwise. One letter will be coloured blue, indicating the
passenger pick-up location, and one letter will be coloured purple, indicating
the passenger drop-off location.

There are several walls placed around the environment. Most are on the sides of
the environment, but there are a few which are located nearer the centre.

Recall that our action space is of size 6. Our reinforcement learning algorithm
will represent each action with a number from 0 to 5.

We hope that our agent will learn to map the right action to any given state.
This is achieved by allowing the agent to explore the enivronment. Our agent's
inital behaviour will probably seem random, but it should develop a keen sense
of the optimal action to take in any given state through the rewards and
punishments it recieves during its time exploring.



Q-learning



Q-learning is how our agent is going to learn the long-term best strategy for
maximising its reward. Firstly, we need a table, known as a Q-table, that
maps all Q-values (equation coming up) of each state and action combination.
The table is of the form state x action, with state representing the rows, and
action representing the columns. Each value in the Q-table is known as a
Q-value.

Q-values are initialised to 0, but are then updated as the agent interacts
with the environment. Each Q-value is updated using the following equation:

Q(state, action) <-- (1 - alpha)Q(state, action) + alpha(reward + gamma x
max Q(next state, all actions)),

where:
1. Alpha is the learning rate, i.e., the extent to which the Q-values are
being updated on each iteration.
2. Gamma is the future reward discount factor, i.e., to what extent do we want
our agent to prioritise future reward over present reward. A high value weighs
towards long-term reward, whereas a low value weighs towards immediate reward.

You can probably see how this leads to the agent optimising for the long term.
By making the Q-value of any state and action combination a weighted sum between
the immediate reward and future reward, it is forced to consider both instead of
just maximising immediate reward, which might not be the best strategy in the
long run. Our agent chooses the Q-value that is highest, and continues to do so
until its goal is complete.

There's a tradeoff between exploration (choosing a random action) and
exploitation (choosing actions based on learned Q-values). We want our agent to
avoid overfitting (exploiting too much) and underfitting (exploring too much),
so we need to introduce a parameter to find a balance between the two. We'll
call this parameter epsilon. Lower epsilon values favour exploring more often,
whereas higher values favour exploiting more often.



Training the agent



'''
>>> import gym
>>> import numpy as np
>>> import random
>>>
>>> q_table = np.zeros([env.observation_space.n, env.action_space.n])
>>> q_table.shape
(500, 6)
'''



Now, we will create the training algorithm that will update the Q-table as our
agent explores the environement over many iterations. The first part of our
algorithm will be to decide whether we want to pick an action according to our
Q-table, or choose a random action. This will be done using the epsilon and the
random.uniform(0, 1) function which returns a random value between 0 and 1.



'''
>>> for i in range(1, 100001):
    
        state = env.reset()
	epochs, penalties, reward = 0, 0, 0
	done = False
	
	while not done:
	    if random.uniform(0, 1) < epsilon:
	        action = env.action_space.sample() # Explore the action space.
	    else:
	        action = np.argmax(q_table[state]) # Exploit learned values.
	        
	    next_state, reward, done, info = env.step(action)
	    old_value = q_table[state, action]
	    
	    next_max = np.max(q_table[next_state])
	    new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
	    q_table[state, action] = new_value
	    
	    if reward == -10:
		penalties += 1
		
	    state = next_state
	    epochs += 1
	    
	if i % 10000 == 0:
            print(f"Episode: {i}")

		
Episode: 10000
Episode: 20000
Episode: 30000
Episode: 40000
Episode: 50000
Episode: 60000
Episode: 70000
Episode: 80000
Episode: 90000
Episode: 100000
'''



Evaluating the agent



Now that our agent has an idea of what it should do in each state, its time to
test how good it is. The agent has had a chance to explore ts options in the
last section, so now we'll focus on choosing the highest Q-value of each state.
i.e., our agent will no longer be exploring; only exploiting.



'''
>>> for i in range(episodes):

	state = env.reset()
	epochs, penalties, rewards = 0, 0, 0
	done = False
	
	while not done:
            
		action = np.argmax(q_table[state])
		state, reward, done, info = env.step(action)
		
		if reward == -10:
			penalties += 1
			
		epochs += 1
		
	total_penalties += penalties
	total_epochs += epochs

>>> print(f"Average penalties per episode: {total_penalties / episodes}")
Average penalties per episode: 0.0
'''



Seeing as the agent averaged no penalties across the 100 episodes, this must
meen that it made the correct decisions at each time step, and has therefore
learned the optimal strategy.



Future improvements



While not particularly relevant here, we would normally adjust the
hyperparameters as we went along, like so:

alpha - As the agent becomes more experienced with time, we should decrease the
learning rate to allow the agent to fine-tune itself more precisely.

Gamma - If we were on a time dependent project, we could lower the future reward
weighting to force our agent to consider more short-term rewards.

Epsilon - Similarly to alpha, once the agent has developed a broad enough
knoweldge base, there is not as much need for it to explore the environment, and
therefore we would favour exploitation.



