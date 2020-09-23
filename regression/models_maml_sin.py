import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from copy import deepcopy
from collections import OrderedDict

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

'''
The few-shot classifier model as used in Reptile.
Toy sinusoidal problem
'''

# Model definition

class SinusoidModel(nn.Module):

    def __init(self):

        super(SinusoidModel, self).__init__()

        self.lin1 = nn.Linear(1, 64)
        self.lin2 = nn.Linear(64, 64)
        self.lin3 = nn.Linear(64, 1)

    def forward(self, x):

        x = F.tanh(self.lin1(x))
        x = F.tanh(self.lin2(x))
        x = self.lin3(x)

        return x

    def functional_forward(self, x, weight_dict):

        x = F.linear(x, weight=weight_dict['lin1.weight'], bias=weight_dict['lin1.bias'])
        x = F.tanh()

        x = F.linear(x, weight=weight_dict['lin2.weight'], bias=weight_dict['lin2.bias'])
        x = F.tanh()

        x = F.linear(x, weight=weight_dict['lin3.weight'], bias=weight_dict['lin3.bias'])

        return x

class taskGenerator:

    def __init__(self, meta_batch_size, train_shot=10, query_shot=None, rng=None):
        
        self.meta_batch_size = meta_batch_size
        self.train_shot = train_shot
        self.query_shot = query_shot

        self.x_sample = meta_batch_size*train_shot
        if query_shot is not None:
            self.x_sample = meta_batch_size*(train_shot+query_shot)
        
        self.x = np.expand_dims(np.linspace(-5, 5, self.x_sample), 1)

    def sample_task(self):

        phase = rng.uniform(low=0, high=2*np.pi)
        ampl = rng.uniform(0.1, 5)
        
        y = lambda x : np.sin(self.x + phase) * ampl
        
        return y


class Reptile:
    """
    A meta-learning session.

    """

    def __init__(self, model, device, update_lr, meta_step_size):

        self.device = device
        self.net = model.to(self.device)
        self.update_optim = optim.SGD(self.net.parameters(), lr=update_lr)
        self.meta_optim = optim.SGD(self.net.parameters(), lr=meta_step_size)

    def train_step(self,
                    taskGen,
                    num_shots,
                    meta_step_size,
                    meta_batch_size):
    
        # Adapt the meta learning rate.
        for param_group in self.meta_optim.param_groups:
            param_group['lr'] = meta_step_size

        # Save state and variables of meta learner.
        old_state = deepcopy(self.net.state_dict())

        old_vars = []
        for params in self.net.parameters():
            old_vars.append(params.cpu())

        new_vars = []

        # Train on meta batch
        for i in range(meta_batch_size):

            inputs, labels = taskGen.x, taskGen.sample_task()
            inputs = (torch.tensor(inputs)).to(self.device)
            labels = (torch.tensor(labels)).to(self.device)

            self.update_optim.zero_grad()
            logits = self.net(inputs)
            loss = F.mse_loss(logits, labels)
            loss.backward()
            self.update_optim.step()
            self.update_optim.zero_grad()

            # Save updated parameters after a task
            update_net_parameters_list = []
            for params in self.net.parameters():
                update_net_parameters_list.append(params.cpu())
            new_vars.append(update_net_parameters_list)

            # Return the meta learner to the state before the task.
            self.net.load_state_dict(old_state)

        self.meta_optim.zero_grad()

        # compute the new gradients.
        # 1. store the sum of new variables in new_vars[0].
        # 2. Find the average parameters.
        # 3. calculate the difference and update the meta learner gradients.

        for i in range(1, len(new_vars)):
            for params1, params2 in zip(new_vars[0], new_vars[i]):
                params1 = params1.add_(params2)

        cumulative_grad = [wt / meta_batch_size for wt in new_vars[0]]

        for counter in range(len(old_vars)):
            cumulative_grad[counter] = torch.sub(
                old_vars[counter], cumulative_grad[counter])

        counter = 0
        for params in self.net.parameters():
            x = cumulative_grad[counter]
            params.grad = x.to(self.device)
            counter = counter+1

        # update the meta learner parameters
        self.meta_optim.step()
        self.meta_optim.zero_grad()

    def evaluate(self,
                all_inputs,
                all_labels,
                test_inputs,
                test_labels,
                inner_iters):

        '''
        save the old state of the meta learner and the optimizer to stop leakage.
        '''
        plt.cla()
        plt.plot(all_inputs, self.predict(all_inputs), label='pred after 0', color=(0, 0, 1))

        old_state = deepcopy(self.net.state_dict())
        update_optim_old_state = deepcopy(self.update_optim.state_dict())

        inputs = (torch.tensor(test_inputs)).to(self.device)
        labels = (torch.tensor(test_labels)).to(self.device)

        for innerIter in range(inner_iters):

            self.update_optim.zero_grad()
            logits = self.net(inputs)
            loss = F.mse_loss(logits, labels)
            loss.backward()
            self.update_optim.step()

            if (innerIter+1) % 8 == 0:
                frac = (innerIter+1) / inner_iters
                plt.plot(all_inputs, self.predict(all_inputs), 
                    label='pred after '+ str(innerIter+1), color=(frac, 0, 1-frac))

        plt.plot(all_inputs, all_labels, label='true', color=(0,1,0))
        
        lossval = np.square(self.predict(all_inputs) - all_labels).mean()
        plt.plot(test_inputs, test_labels, 'x', label='train', color='k')
        
        plt.ylim(-4,4)
        plt.legend(loc='lower right')
        plt.pause(0.01)
        
        print('-----------------------------')
        print('iteration '+str(innerIter+1))
        print('loss on plotted curve'+ str(lossval))

        self.net.load_state_dict(old_state)
        self.update_optim.load_state_dict(update_optim_old_state)

        return 

    def predict(self, inputs):

        inputs = (torch.tensor(inputs)).to(self.device)
        predictions = self.net(inputs).cpu()
        prediction = predict.data.numpy()

        return prediction
        

def main():

    seed = 0
    update_lr = 0.02  
    meta_step_size = 0.1  
    meta_iters = 3000
    meta_batch_size = 1
    inner_iters = 32

    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)

    taskGen = taskGenerator(meta_batch_size=meta_batch_size, 
        train_shot=train_shot, query_shot=None, rng=rng)
    
    all_inputs = taskGen.x_sample
    all_labels = taskGen.sample_task()

    test_indices = rng.choice(len(all_inputs), size=train_shot)
    test_inputs = all_inputs[test_indices]
    test_labels = all_labels[test_indices]

    model = SinusoidModel()
    device = torch.device('cuda')
    reptile_model = Reptile(model, device, update_lr, meta_step_size)
    
    for current_iter in range(meta_iters):
        
        frac = current_iter / meta_iters
        current_step_size = meta_step_size * (1 - frac)
        reptile_model.train(taskGen, train_shot, current_step_size, meta_batch_size)
    
    reptile_model.evaluate(all_inputs, all_labels, test_inputs, test_labels, inner_iters)

if __name__ == '__main__':
    main()











    




