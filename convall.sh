#!/bin/bash

for file in $1/*
do
	./kmz2mc.py --model "$file" --world "$2"
done
