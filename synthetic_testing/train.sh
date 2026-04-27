#!/bin/bash

#first argument is the method and the second argument is the kernel size
if [ "$#" -lt 2 ]
then
	echo "Usage: $0 <method> <kernel_size>" >&2
	exit 1
fi

if [ "$1" = "analytical" ]
then
	python train_new.py --kernel_size "$2"
else
	python train_new.py --method uniform --kernel_size "$2"
fi
