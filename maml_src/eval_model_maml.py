"""
Helpers for evaluating models.
"""
import numpy as np
from .maml import MAML

def evaluate(maml_model,
            dataset,
            num_classes=5,
            num_shots=5,
            eval_inner_iters=10,
            replacement=False,
            num_samples=10000):
    """
    Evaluate a model on a dataset.
    """

    total_correct = []
    for _ in range(num_samples):
        total_correct.append(maml_model.evaluate(dataset, 
                        num_classes=num_classes, num_shots=num_shots,
                        inner_iters=eval_inner_iters, replacement=replacement))
    
    total_accuracies = np.array(total_correct) / num_classes
    test_accuracy = total_accuracies.sum() / num_samples
    test_variation = np.std(total_accuracies)

    return total_accuracies, test_accuracy, test_variation
