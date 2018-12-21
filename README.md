# SEU_NNFailure
This project tests Neural Networks' robustness against Single Event Upset.

## Workflow

A pretrained Neural Networks, invert one of its weights' first bit, test the disrupted network on a validation set, then record the accuracy loss. Test every bits (or randomly sample some bits) of this network, analyze the effect of their inversion.

## Experiments

Experiments are conducted mostly on  [XNOR-Net](https://github.com/allenai/XNOR-Net)

### XNOR-Net ImpeImplementation

Used an [PyTorch implementation](https://github.com/jiecaoyu/XNOR-Net-PyTorch) of the [XNOR-Net](https://github.com/allenai/XNOR-Net). Major networks are as follows:

| Dataset  | Network                  | Accuracy                    |
| -------- | :----------------------- | :-------------------------- |
| MNIST    | LeNet-5                  | 99.23%                      |
| CIFAR-10 | Network-in-Network (NIN) | 86.28%                      |
| ImageNet | AlexNet                  | Top-1: 44.87% Top-5: 69.70% |

### MNIST

### CIFAR-10

### ImageNet