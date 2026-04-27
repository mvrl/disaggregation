#!/bin/bash

#Uniform
python train_new.py --method uniform --kernel_size 1
python train_new.py --method uniform --kernel_size 4
python train_new.py --method uniform --kernel_size 8
python train_new.py --method uniform --kernel_size 16

#Analytical
python train_new.py --method analytical --kernel_size 1
python train_new.py --method analytical --kernel_size 4
python train_new.py --method analytical --kernel_size 8
python train_new.py --method analytical --kernel_size 16