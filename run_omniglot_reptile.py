"""
Train a model on miniImageNet.
"""

import random
import os
import sys
import numpy as np

import torch

import argparse
from functools import partial
from omniglot import read_dataset, split_dataset, augment_dataset

from reptile_src.models_reptile import OmniglotModel
from reptile_src.eval_model_reptile import evaluate
from reptile_src.train_reptile import train
from reptile_src.reptile import Reptile

DATA_DIR = 'data/omniglot'

def argument_parser():
    """
    Get an argument parser for a training script.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', help='random seed', default=0, type=int)
    parser.add_argument('--classes', help='number of classes per inner task', default=5, type=int)
    parser.add_argument('--shots', help='number of examples per class', default=5, type=int)
    parser.add_argument('--train-shots', help='shots in a training batch', default=0, type=int)
    parser.add_argument('--inner-batch', help='inner batch size', default=10, type=int)
    parser.add_argument('--inner-iters', help='inner iterations', default=5, type=int)
    parser.add_argument('--replacement', help='sample with replacement', action='store_true')
    parser.add_argument('--learning-rate',help='Adam step size', default=1e-3, type=float)
    parser.add_argument('--meta-step', help='meta-training step size', default=1, type=float)
    parser.add_argument('--meta-batch', help='meta-training batch size', default=5, type=int)
    parser.add_argument('--meta-iters', help='meta-training iterations', default=400000, type=int)
    parser.add_argument('--eval-batch', help='eval inner batch size', default=5, type=int)
    parser.add_argument('--eval-iters', help='eval inner iterations', default=50, type=int)
    parser.add_argument('--eval-samples', help='evaluation samples', default=10000, type=int)
    parser.add_argument('--eval-interval',help='train steps per eval', default=1000, type=int)
    parser.add_argument('--eval-interval-sample',help='evaluation samples during training', default=100, type=int)
    parser.add_argument('--only-evaluation', help='Set for only evaluation', action='store_true', default=False)
    parser.add_argument('--checkpoint', help='Load saved checkpoint from path', default=None)
    return parser

def model_kwargs(parsed_args):

    return {
        'update_lr': parsed_args.learning_rate,
        'meta_step_size': parsed_args.meta_step
    }

def train_kwargs(parsed_args):
    """
    Build kwargs for the train() function from the parsed
    command-line arguments.
    """
    return {
        'num_classes': parsed_args.classes,
        'num_shots': parsed_args.shots,
        'train_shots': (parsed_args.train_shots or None),
        'inner_batch_size': parsed_args.inner_batch,
        'inner_iters': parsed_args.inner_iters,
        'replacement': parsed_args.replacement,
        'meta_step_size': parsed_args.meta_step,
        'meta_batch_size': parsed_args.meta_batch,
        'meta_iters': parsed_args.meta_iters,
        'eval_inner_batch_size': parsed_args.eval_batch,
        'eval_inner_iters': parsed_args.eval_iters,
        'eval_interval': parsed_args.eval_interval,
        'eval_interval_sample': parsed_args.eval_interval_sample
    }

def evaluate_kwargs(parsed_args):
    """
    Build kwargs for the evaluate() function from the
    parsed command-line arguments.
    """
    return {
        'num_classes': parsed_args.classes,
        'num_shots': parsed_args.shots,
        'eval_inner_batch_size': parsed_args.eval_batch,
        'eval_inner_iters': parsed_args.eval_iters,
        'replacement': parsed_args.replacement,
        'num_samples': parsed_args.eval_samples
    }

def main():
    """
    Load data and train a model on it.
    """
    
    args = argument_parser().parse_args()
    torch.manual_seed(args.seed)

    fileStart = 'omniglot_reptile_checkpoint_seed_' + str(args.seed)
    if not os.path.exists(fileStart):
        os.makedirs(fileStart)
    
    # model_save_path='model_checkpoint_' + str(args.seed) + '.pt'
        
    device = torch.device('cuda')

    train_set, test_set = split_dataset(read_dataset(DATA_DIR))
    train_set = list(augment_dataset(train_set))
    test_set = list(test_set)
    val_set = None

    model = OmniglotModel(args.classes)

    reptile_model = Reptile(model, device, **model_kwargs(args))
    
    if args.only_evaluation is True:
        
        assert args.checkpoint is not None, 'For evaluating without training please provide a checkpoint'
        
        checkpoint_model = torch.load(args.checkpoint)
        reptile_model.net.load_state_dict(checkpoint_model['model_state'])
        reptile_model.meta_optim.load_state_dict(checkpoint_model['meta_optim_state'])
        reptile_model.update_optim.load_state_dict(checkpoint_model['update_optim_state'])
    
    else:

        save_path = fileStart + '/' + 'final_model.pt'

        train(reptile_model, train_set, val_set, test_set, fileStart, **train_kwargs(args))
        torch.save({'model_state': reptile_model.net.state_dict(),
                    'update_optim_state': reptile_model.update_optim.state_dict(),
                    'meta_optim_state': reptile_model.meta_optim.state_dict()},
                    save_path) 

    print('Evaluating...')
    eval_kwargs = evaluate_kwargs(args)

    train_accuracies, train_accuracy, train_variation = evaluate(reptile_model, train_set, **eval_kwargs)
    test_accuracies, test_accuracy, test_variation = evaluate(reptile_model, test_set, **eval_kwargs)

    print('Train accuracy: ' + str(train_accuracy) + '; Train variation: ' + str(train_variation))
    print('Test accuracy: ' + str(test_accuracy) + '; Test variation: ' + str(test_variation))

    save_path = fileStart + '/' + 'results.npz'
    np.savez(save_path, train_accuracies=train_accuracies, test_accuracies=test_accuracies)

if __name__ == '__main__':
    main()
