#!/bin/bash

#first argument is the method and the second argument is the kernel size
if [ $1 == 'analytical' ]
then
	python train_small.py --kernel_size $2
else
	python train_small.py --method uniform --kernel_size $2
fi
