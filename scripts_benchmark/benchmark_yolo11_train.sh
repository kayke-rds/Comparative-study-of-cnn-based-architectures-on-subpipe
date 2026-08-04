#!/bin/bash

mkdir -p ./benchmark_data

nvidia-smi --query-gpu=timestamp,power.draw,temperature.gpu,utilization.gpu,memory.used \
	--format=csv -l 1 -f ./benchmark_data/energia_gpu-yolo11-train.csv &

PID_NVIDIA=$!

perf stat -e page-faults,context-switches,cpu-migrations,L1-dcache-loads,L1-dcache-load-misses,L1-icache-loads,LLC-loads,LLC-load-misses -x ';' \
	-o ./benchmark_data/dados_benchmark_yolo11_train.csv python3 ./model-yolo11-subpipemini.py

kill $PID_NVIDIA
