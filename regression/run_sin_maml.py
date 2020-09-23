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
from miniimagenet import read_dataset

from maml_src.models_maml import MiniImageNetModel
from maml_src.eval_model_maml import evaluate
from maml_src.train_maml import train
from maml_src.maml import MAML

DATA_DIR = 'data/miniimagenet'

def argument_parser():
    """
    Get an argument parser for a training script.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', help='random seed', default=0, type=int)
    parser.add_argument('--order', help='MAML order can be 2 or 1', default=2, type=int)
    parser.add_argument('--classes', help='number of classes per inner task', default=5, type=int)
    parser.add_argument('--shots', help='number of examples per class', default=1, type=int)
    parser.add_argument('--meta-shots', help='shots for meta update', default=5, type=int)
    parser.add_argument('--inner-iters', help='inner iterations', default=5, type=int)
    parser.add_argument('--replacement', help='sample with replacement', action='store_true', default=False)
    parser.add_argument('--learning-rate',help='base learner update step size', default=0.01, type=float)
    parser.add_argument('--meta-step', help='meta-learner Adam step size', default=0.001, type=float)
    parser.add_argument('--meta-batch', help='meta-training batch size', default=4, type=int)
    parser.add_argument('--meta-iters', help='meta-training iterations', default=60000, type=int)
    parser.add_argument('--eval-iters', help='eval inner iterations', default=10, type=int)
    parser.add_argument('--eval-samples', help='evaluation samples', default=600, type=int)
    parser.add_argument('--eval-interval',help='Evaluation interval during training', default=1000, type=int)
    parser.add_argument('--eval-interval-sample',help='evaluation samples during training', default=600, type=int)
    parser.add_argument('--only_evaluation', help='Set for only evaluation', action='store_true', default=False)
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
        'order': parsed_args.order,
        'num_classes': parsed_args.classes,
        'num_shots': parsed_args.shots,
        'meta_shots': parsed_args.meta_shots,
        'inner_iters': parsed_args.inner_iters,
        'replacement': parsed_args.replacement,
        'meta_batch_size': parsed_args.meta_batch,
        'meta_iters': parsed_args.meta_iters,
        'eval_inner_iters': parsed_args.eval_iters,
        'eval_interval': parsed_args.eval_interval,
        'num_samples': parsed_args.eval_samples,
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
        'eval_inner_iters': parsed_args.eval_iters,
        'replacement': parsed_args.replacement,
        'num_samples': parsed_args.eval_samples
    }

def main():
    """
    Load data and train a model on it.
    """
    sys.dont_write_bytecode = True
    
    args = argument_parser().parse_args()
    torch.manual_seed(args.seed)

    fileStart = 'miniimagenet_maml_checkpoint_' + str(args.order) + '_' + str(args.seed)
    if not os.path.exists(fileStart):
        os.makedirs(fileStart)

    device = torch.device('cuda')

    train_set, val_set, test_set = read_dataset(DATA_DIR)

    model=MiniImageNetModel(args.classes)

    maml_model = MAML(model, device, **model_kwargs(args))

    if args.checkpoint is not None:
        checkpoint_model = torch.load(args.checkpoint)
        maml_model.net.load_state_dict(checkpoint_model['model_state'])
        maml_model.meta_optim.load_state_dict(checkpoint_model['meta_optim_state'])

    if args.only_evaluation is True:
        
        if args.checkpoint is None:
            sys.exit('For evaluating without training please provide a checkpoint')

    else:
        train(maml_model, train_set, val_set, test_set, fileStart, **train_kwargs(args))

        save_path = fileStart + '/' + 'final_model.pt'

        torch.save({'model_state': maml_model.net.state_dict(),
                    'meta_optim_state': maml_model.meta_optim.state_dict()},
                    save_path)

    print('Evaluating...')
    eval_kwargs = evaluate_kwargs(args)

    train_accuracies, train_accuracy, train_variation = evaluate(maml_model, train_set, **eval_kwargs)
    val_accuracies, val_accuracy, val_variation = evaluate(maml_model, val_set, **eval_kwargs)
    test_accuracies, test_accuracy, test_variation = evaluate(maml_model, test_set, **eval_kwargs)

    print('Train accuracy: ' + str(train_accuracy) + '; Train variation: ' + str(train_variation))
    print('Validation accuracy: ' + str(val_accuracy) + '; Validation variation: ' + str(val_variation))
    print('Test accuracy: ' + str(test_accuracy) + '; Test variation: ' + str(test_variation))

    save_path = fileStart + '/' + 'results.npz'
    np.savez(save_path, train_accuracies=train_accuracies, val_accuracies=val_accuracies, 
        test_accuracies=test_accuracies)

if __name__ == '__main__':
    main()
