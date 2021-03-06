import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_s(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

def conv_dw(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )

def state_model():
    return nn.Sequential(
            # CHW: 1, 32, 32
            conv_s(  1,  32, 1), # 32, 32, 32
            conv_dw( 32,  32, 1), # 32, 32, 32
            conv_dw( 32,  32, 2), # 32, 16, 16
            conv_dw( 32,  16, 1), # 16, 16, 16
            conv_dw( 16,  16, 2), # 16,  8,  8
            nn.AvgPool2d(8),      # 16,  1,  1
        )

# Creating the architecture of the Neural Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.model = state_model()
        self.fc1 = nn.Linear(16, 100)
        self.fc2 = nn.Linear(100, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 16)
        x = self.fc2(self.fc1(x))
        x = self.max_action * torch.tanh(x)
        return x

# Creating the architecture of the Neural Network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Defining the first Critic neural network
        self.c1_model = state_model()
        self.c1_fc1 = nn.Linear(16 + action_dim, 100)
        self.c1_fc2 = nn.Linear(100, 1)
        # Defining the second Critic neural network
        self.c2_model = state_model()
        self.c2_fc1 = nn.Linear(16 + action_dim, 100)
        self.c2_fc2 = nn.Linear(100, 1)

    def forward(self, x, u):
      #xu = torch.cat([x, u], 2)
      # Forwad-Propagation on the first Critic Neural Network
      x1 = self.c1_model(x)
      x1 = x1.view(-1, 16)
      x1 = torch.cat([x1, u], 1)
      x1 = self.c1_fc2(self.c1_fc1(x1))
      # Forward-Propagation on the second Critic Neural Network
      x2 = self.c2_model(x)
      x2 = x2.view(-1, 16)
      x2 = torch.cat([x2, u], 1)
      x2 = self.c2_fc2(self.c2_fc1(x2))
      return x1, x2

    def Q1(self, x, u):
      x1 = self.c1_model(x)
      x1 = x1.view(-1, 16)
      x1 = torch.cat([x1, u], 1)
      x1 = self.c1_fc2(self.c1_fc1(x1))
      return x1

# Building the whole Training Process into a class
class T3D(object):
  
  def __init__(self, state_dim, action_dim, max_action):
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.max_action = torch.Tensor(max_action).to(self.device)
    self.actor = Actor(state_dim, action_dim, self.max_action).to(self.device)
    self.actor_target = Actor(state_dim, action_dim, self.max_action).to(self.device)
    self.actor_target.load_state_dict(self.actor.state_dict())
    self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
    self.critic = Critic(state_dim, action_dim).to(self.device)
    self.critic_target = Critic(state_dim, action_dim).to(self.device)
    self.critic_target.load_state_dict(self.critic.state_dict())
    self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
    
  def select_action(self, state):
    state = state.unsqueeze(0).to(self.device)
    return self.actor(state).cpu().data.numpy().flatten()

  def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
    for it in range(iterations):
      # We sample a batch of transitions (s, s', a, r) from the memory
      batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = replay_buffer.sample(batch_size)
      state = torch.Tensor(batch_states).to(self.device)
      next_state = torch.Tensor(batch_next_states).to(self.device)
      action = torch.Tensor(batch_actions).to(self.device)
      reward = torch.Tensor(batch_rewards).to(self.device)
      done = torch.Tensor(batch_dones).to(self.device)
      
      # From the next state s', the Actor target plays the next action a'
      next_action = self.actor_target(next_state)#.squeeze(1)

      # We add Gaussian noise to this next action a' and we clamp it in a range of values supported by the environment
      noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(self.device)
      noise = noise.clamp(-noise_clip, noise_clip)
      next_action = (next_action + noise).clamp(-self.max_action[0], self.max_action[0])

      # The two Critic targets take each the couple (s', a') as input and return two Q-values Qt1(s',a') and Qt2(s',a') as outputs
      target_Q1, target_Q2 = self.critic_target(next_state, next_action)
      
      # We keep the minimum of these two Q-values: min(Qt1, Qt2)
      target_Q = torch.min(target_Q1, target_Q2)
      
      # We get the final target of the two Critic models, which is: Qt = r + gamma * min(Qt1, Qt2), where gamma is the discount factor
      target_Q = reward + ((1 - done) * discount * target_Q).detach()
      
      # The two Critic models take each the couple (s, a) as input and return two Q-values Q1(s,a) and Q2(s,a) as outputs
      current_Q1, current_Q2 = self.critic(state, action)
      
      # We compute the loss coming from the two Critic models: Critic Loss = MSE_Loss(Q1(s,a), Qt) + MSE_Loss(Q2(s,a), Qt)
      critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
      
      # We backpropagate this Critic loss and update the parameters of the two Critic models with a SGD optimizer
      self.critic_optimizer.zero_grad()
      critic_loss.backward()
      self.critic_optimizer.step()
      
      # Once every two iterations, we update our Actor model by performing gradient ascent on the output of the first Critic model
      if it % policy_freq == 0:
        actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Still once every two iterations, we update the weights of the Actor target by polyak averaging
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
          target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
        # Still once every two iterations, we update the weights of the Critic target by polyak averaging
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
          target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
  
  # Making a save method to save a trained model
  def save(self, filename, directory):
    torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
    torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))
  
  # Making a load method to load a pre-trained model
  def load(self, filename, directory):
    self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename), map_location=torch.device('cpu')))
    self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename), map_location=torch.device('cpu')))
