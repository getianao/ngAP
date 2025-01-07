ngap -a /home/tge/workspace/ngAP/automata_benchmark_original/AutomataZoo/Brill/benchmarks/anml_remove_or/automata_0.anml -i /home/tge/workspace/ngAP/automata_benchmark_original/AutomataZoo/Brill/benchmarks/inputs/brown_corpus.txt --app-name=Brill --algorithm=nonblockingallgroups --input-start-pos=0 --input-len=1000000 --split-entire-inputstream-to-chunk-size=1000000 --group-num=1 --duplicate-input-stream=600 --unique=false --unique-frequency=10 --use-soa=false --result-capacity=54619400 --use-uvm=false --data-buffer-fetch-size=25600 --add-aan-start=256 --add-aas-interval=1000000 --active-threshold=0 --precompute-cutoff=-1 --precompute-depth=3 --compress-prec-table=true --pc-use-uvm=false --report-off=false --remove-degree=false --quit-degree=false --max-nfa-size=-1 --adaptive-aas=true --quick-validation=9038877 --validation=true

## 3090

Run
``` bash
# Throughput
time ${NGAP_ROOT}/scripts/run-throughput.sh
# Latency
time ${NGAP_ROOT}/scripts/run-latency.sh
# Breakdown
# time ${NGAP_ROOT}/scripts/run-breakdown.sh
time ${NGAP_ROOT}/scripts/run_throughput_NAP_breakdown.sh
# Profile
time ${NGAP_ROOT}/scripts/run_ncu.sh
```

Plot
``` bash
# Throughput
${NGAP_ROOT}/scripts/gen-throughput-fig13tab4.sh
# Latency
${NGAP_ROOT}/scripts/gen-latency-fig20tab6.sh
# Breakdown
python ${NGAP_ROOT}/scripts/plot_throughput_gpu_nap_breakdown.py
# Profile
python ${NGAP_ROOT}/scripts/plot_ncu_memory_stack.py
python ${NGAP_ROOT}/scripts/plot_ncu.py
```


## V100

Run
``` bash
# Throughput
time ${NGAP_ROOT}/scripts/run-throughput.sh
```

Plot
``` bash
# Throughput

# Roofline

# vs3090
```
