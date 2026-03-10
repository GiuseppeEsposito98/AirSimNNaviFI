#!/bin/bash

active_processes=$(ps -ef | grep Block | tr -s ' ' | cut -d ' ' -f2)
echo ${active_processes}
arraysize=${#active_processes[@]}

for ((i=0; i<arraysize; i++)); do
    kill ${active_processes[$((i))]}
done