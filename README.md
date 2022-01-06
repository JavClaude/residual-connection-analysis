# resco

## Description

resco is a simple python package that report the effect of [deep residual learning](https://arxiv.org/abs/1512.03385) on backpropagation for 2 neural network architectures:
- A plain linear neural network 
- A residual linear neural network

## Installation

Install and create a virtualenv

```bash
pip install virtualenv
virtualenv .venv
```

Install the package
```
pip install . 
```

## Execution
```bash
resco_analysis --model_name plain_dnn --n_blocks 50 --lr 0.001 --batch_size 256 --n_epochs 5
```


Gradients of the first neural network layer are reported on tensorboard, you can run your own tensorboard server by using this command
```bash
docker image build --tag tensorboard_resco .
docker container run --restart always -d -p 5001:5001 tensorboard_lm
```
