#!/bin/bash

classes=("traversable area" "walkable area" "runnable area" "road" "way" "walkway" "sidewalk" "street" "pavement")

for class in "${classes[@]}"
do
    python3 grounded_sam_simple_demo.py --classes "$class"
done