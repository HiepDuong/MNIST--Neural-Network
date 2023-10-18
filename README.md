# MNIST--Neural-Network

## Introduction

The MNIST dataset is a popular benchmark for evaluating and understanding machine learning models for digit recognition. Here, we document the evolution of models and their performances as they are enhanced and refined.

## MLP model
- First model: using default parameter
- Second model: optimized by tweaking hyperparameters: with dropout, learning rate, batchsize, early stopping

# MNIST Model Evolution with CNN

Documents the progression and evolution of models trained on the MNIST dataset.


## Models Overview

| Model | Description                                                                                                                                                                   | Training Approach                                                                                                                             | Accuracy |
|-------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------|----------|
| **Model 1** | 3 MLP layers                                                                                                                                                                 | Standard Optimization                                                                                                                         | 98%      |
| **CNN 0.0** | 3 CNN layers with max pooling followed by 2 MLP layers                                                                                                                       | Adam optimizer with `lr=0.001` and `torch.optim.lr_scheduler.ReduceLROnPlateau` with a threshold of 0.01                                       | 98-99%   |
| **CNN 1**   | As in CNN 0.0 but adding 1 extra MLP layer                                                                                                                                    | Adam optimizer with scheduler as `torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)` and early stopping with patience of 10 | 99.38%   |
| **CNN 2**   | Same architecture as CNN 1                                                                                                                                                   | Adam optimizer with `lr=0.0005` and early stopping with patience of 30                                                                         | 99.46%   |
| **CNN 3**   | Same architecture as CNN 2                                                                                                                                                   | Adam optimizer with `lr=0.001`                                                                                                                 | 99.62%   |
| **CNN 4**   | Addition of 1 more CNN layer compared to CNN 3, and gamma set to 0.9                                                                                                         | Standard Optimization                                                                                                                         | 99.73%   |

## Detailed Descriptions

### Model 1: MLP 
- **Architecture**: Comprises 3 MLP layers.
- **Results**: Achieved a test accuracy of 98%.

### CNN 0.0
- **Architecture**: 
  - 2-3 CNN layers paired with max pooling.
  - 2 MLP layers.
- **Optimization**:
  - Optimizer: Adam (`lr=0.001`)
  - Scheduler: `torch.optim.lr_scheduler.ReduceLROnPlateau` with a threshold of 0.01.
- **Results**: Accuracy ranges between 98% and 99%.

### CNN 1
- **Architecture**: 
  - Similar to CNN 0.0 but with an additional MLP layer.
- **Optimization**:
  - Optimizer: Adam (lr=0.0005)
  - Scheduler: `torch.optim.lr_scheduler.MultiStepLR` with specified milestones and gamma.
  - Implemented early stopping with a patience of 10.
- **Results**: Achieved a test accuracy of 99.38%.

### CNN 2
- **Architecture**: 
  - Identical to CNN 1.
- **Optimization**:
  - Early stopping implemented with a patience of 30.
- **Results**: Achieved a test accuracy of 99.46%.

### CNN 3
- **Architecture**: 
  - Identical to CNN 2.
- **Optimization**:
  - Optimizer: Adam (`lr=0.001`)
- **Early Stopping**:
  - Early stopping was implemented based on validation loss, but the resulting model was not used. Instead, a model with the minimum `val_loss` was chosen.
- **Results**: Achieved a test accuracy of 99.62%.

### CNN 4
- **Architecture**: 
  - One more CNN layer added compared to CNN 3.
- **Optimization**:
  - Gamma set to 0.9.
- **Model Selection Criteria**:
  - Model checkpoint saved when `val_loss` was at its minimum.
  - Although the model was trained for 400 epochs, the performance at the end was worse than the model at the minimum `val_loss`. Therefore, the checkpoint with the minimum `val_loss` was chosen.
- **Early Stopping**: 
  - Not implemented for this model.
- **Results**: Achieved a test accuracy of 99.73%.

## Conclusion

The progression of models showcases a steady increase in accuracy as architectural changes and optimization strategies are implemented. The combination of CNN layers with varied learning rates, schedulers, and early stopping parameters has proven to be effective in enhancing model performance.

## References

1. LeCun, Y., Cortes, C., & Burges, C. J. (2010). MNIST handwritten digit database.
