"""
Training helpers for supervised meta-learning.
"""

import os
import numpy as np

import torch
from .reptile import Reptile

def train(reptile_model,
        train_set,
        val_set,
        test_set,
        model_save_path=None,
        num_classes=5,
        num_shots=5,
        train_shots=None,
        inner_batch_size=5,
        inner_iters=20,
        replacement=False,
        meta_step_size=0.1,
        meta_batch_size=1,
        meta_iters=400000,
        eval_inner_batch_size=5,
        eval_inner_iters=50,
        eval_interval=1000,
        eval_interval_sample=100):
    
    """
    Train a model on a dataset.
    """
    train_accuracy, test_accuracy = [], []

    if val_set is not None:
        val_accuracy = []

    for i in range(meta_iters):
        
        frac_done = i / meta_iters
        cur_meta_step_size = (1 - frac_done) * meta_step_size

        reptile_model.train_step(train_set,  
                        num_classes=num_classes, num_shots=(train_shots or num_shots),
                        inner_batch_size=inner_batch_size, inner_iters=inner_iters,
                        replacement=replacement,
                        meta_step_size=cur_meta_step_size, meta_batch_size=meta_batch_size)
        
        if i % eval_interval == 0:
            
            total_correct = 0
            for _ in range(eval_interval_sample):
                total_correct += reptile_model.evaluate(train_set,
                                    num_classes=num_classes, num_shots=num_shots,
                                    inner_batch_size=eval_inner_batch_size,
                                    inner_iters=eval_inner_iters, replacement=replacement)

            train_accuracy.append(total_correct / (eval_interval_sample * num_classes))

            total_correct = 0
            for _ in range(eval_interval_sample):
                total_correct += reptile_model.evaluate(test_set,
                                    num_classes=num_classes, num_shots=num_shots,
                                    inner_batch_size=eval_inner_batch_size,
                                    inner_iters=eval_inner_iters, replacement=replacement)

            test_accuracy.append(total_correct / (eval_interval_sample * num_classes))

            save_path = model_save_path + '/intermediate_' + str(i) + '_model.pt'

            torch.save({'model_state': reptile_model.net.state_dict(),
                        'update_optim_state': reptile_model.update_optim.state_dict(),
                        'meta_optim_state': reptile_model.meta_optim.state_dict()},
                       save_path)

            if val_set is not None:
                total_correct = 0
                for _ in range(eval_interval_sample):
                    total_correct += reptile_model.evaluate(val_set,
                                        num_classes=num_classes, num_shots=num_shots,
                                        inner_batch_size=eval_inner_batch_size,
                                        inner_iters=eval_inner_iters, replacement=replacement)
                
                val_accuracy.append(total_correct / (eval_interval_sample * num_classes))

                print('batch %d: train=%f val=%f test=%f' % (i,
                    train_accuracy[-1], val_accuracy[-1], test_accuracy[-1]))
            else:
                print('batch %d: train=%f test=%f' % (i,
                    train_accuracy[-1], test_accuracy[-1]))

    res_save_path = model_save_path + '/' + 'intermediate_accuracies.npz'

    if val_set is not None:
        np.savez(res_save_path, train_accuracy=np.array(train_accuracy),
            val_accuracy=np.array(val_accuracy), test_accuracy=np.array(test_accuracy))
    else:
        np.savez(res_save_path, train_accuracy=np.array(train_accuracy),
            test_accuracy=np.array(test_accuracy))

