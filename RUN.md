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
# ${NGAP_ROOT}/scripts/gen-throughput-fig13tab4.sh
python ${NGAP_ROOT}/scripts/plot_throughput_gpu_sota.py
python ${NGAP_ROOT}/scripts/table_throughput.py
# Latency
# ${NGAP_ROOT}/scripts/gen-latency-fig20tab6.sh
python ${NGAP_ROOT}/scripts/plot_throughput_gpu_sota_oneinput.py
python ${NGAP_ROOT}/scripts/table_throughput_oneinput.py
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
python ${NGAP_ROOT}/scripts/plot_throughput_gpu_sota_v100.py
# Roofline

# vs3090
python ${NGAP_ROOT}/scripts/plot_throughput_gpu_ngap_v100_3090.py
```



``` bash
# Fix smallCAV for asyncap
asyncap -a /home/tge/workspace/ngAP/automata_benchmark_original/AutomataZoo/ClamAV/benchmarks/anml_remove_or/automata_0.anml -i /home/tge/workspace/ngAP/automata_benchmark_original/AutomataZoo/ClamAV/benchmarks/inputs/clamav.input --app-name=smallClamAV  --algorithm=runahead  --input-len=1000000  --report-off=true  --duplicate-input-stream=600  --one-output-capacity=10461940  --scanning-R=999999999  --block-size=128  --record-ir=0  --blockDimX=-1  --num-streams=4  --merge-cc=4  --shrmem-wl=1  --shr_wl_len=4  --remove-degree=true  --quit-degree=false --quick-validation=1 --validation=false
```


<!-- time ${NGAP_ROOT}/scripts/run_throughput_gpu_nap_defalut_e2.sh; time ${NGAP_ROOT}/scripts/run_throughput_gpu_nap_best_e2.sh; time ${NGAP_ROOT}/scripts/run-latency.sh; time ${NGAP_ROOT}/scripts/run_ncu.sh -->



## Fix data

### Throughput

RF: best 6.077220 6.517640 ->6.62256
ngap -a /home/tge/workspace/ngAP/automata_benchmark_original/AutomataZoo/RandomForest/benchmarks/anml_remove_or/automata_0.anml -i /home/tge/workspace/ngAP/automata_benchmark_original/AutomataZoo/RandomForest/benchmarks/inputs/20_400_200_inputs/input_features_large.bin --app-name=RandomForest_20_400_200 --algorithm=nonblockingalle2groups --input-start-pos=0 --input-len=1000000 --split-entire-inputstream-to-chunk-size=1000000 --group-num=1 --duplicate-input-stream=600 --unique=false --unique-frequency=10 --use-soa=false --result-capacity=54619400 --use-uvm=false --data-buffer-fetch-size=25600 --add-aan-start=256 --add-aas-interval=1000000 --active-threshold=0 --precompute-cutoff=-1 --precompute-depth=3 --compress-prec-table=true --pc-use-uvm=false --report-off=false --remove-degree=false --quit-degree=false --max-nfa-size=-1 --adaptive-aas=true --use-unique-matchset=true --remove-loop-edge=true --loop-state-prefetch=false --quick-validation=0 --validation=true 


Snort': best 82.974197 85.986198 -> 109.9
ngap -a /home/tge/workspace/ngAP/automata_benchmark_original/AutomataZoo/Snort/benchmarks/anml_remove_or/automata_0.anml -i /home/tge/workspace/ngAP/automata_benchmark_original/AutomataZoo/Snort/benchmarks/inputs/wrccdc2012.pcap --app-name=smallSnort --algorithm=nonblockingalle2groups --input-start-pos=0 --input-len=1000000 --split-entire-inputstream-to-chunk-size=1000000 --group-num=1 --duplicate-input-stream=600 --unique=true --unique-frequency=10 --use-soa=false --result-capacity=54619400 --use-uvm=false --data-buffer-fetch-size=25600 --add-aan-start=256000 --add-aas-interval=256000 --active-threshold=0 --precompute-cutoff=-1 --precompute-depth=3 --compress-prec-table=true --pc-use-uvm=false --report-off=false --remove-degree=true --quit-degree=false --max-nfa-size=-1 --adaptive-aas=false --use-unique-matchset=true --remove-loop-edge=true --loop-state-prefetch=false --quick-validation=128259 --validation=false 

CAV: 
AsyncAP: 103.019638 -> 6.169483
asyncap -a /home/tge/workspace/ngAP/automata_benchmark_original/AutomataZoo/ClamAV/benchmarks/anml_remove_or/automata_0.anml -i /home/tge/workspace/ngAP/automata_benchmark_original/AutomataZoo/ClamAV/benchmarks/inputs/clamav.input --app-name=ClamAV --algorithm=runahead --input-len=1000000 --report-off=true --duplicate-input-stream=600 --one-output-capacity=10461940 --scanning-R=999999999 --block-size=128 --record-ir=0 --blockDimX=-1 --num-streams=4 --merge-cc=4 --shrmem-wl=1 --shr_wl_len=4 --remove-degree=false --quit-degree=false --quick-validation=1 --validation=false 

APR
default: 7.27725 -> 8.45195
ngap -a /home/tge/workspace/ngAP/automata_benchmark_original/AutomataZoo/APPRNG/benchmarks/4_sided/anml_remove_or/automata_0.anml -i /home/tge/workspace/ngAP/automata_benchmark_original/AutomataZoo/APPRNG/benchmarks/4_sided/inputs/10MB_A.prng --app-name=APPRNG4 --algorithm=nonblockingallgroups --input-start-pos=0 --input-len=1000000 --split-entire-inputstream-to-chunk-size=1000000 --group-num=1 --duplicate-input-stream=600 --unique=false --unique-frequency=10 --use-soa=false --result-capacity=54619400 --use-uvm=false --data-buffer-fetch-size=25600 --add-aan-start=256 --add-aas-interval=1000000 --active-threshold=0 --precompute-cutoff=-1 --precompute-depth=3 --compress-prec-table=true --pc-use-uvm=false --report-off=false --remove-degree=false --quit-degree=false --max-nfa-size=-1 --adaptive-aas=true --quick-validation=500000000 --validation=true 

best: 7.647210 -> 9.41817
ngap -a /home/tge/workspace/ngAP/automata_benchmark_original/AutomataZoo/APPRNG/benchmarks/4_sided/anml_remove_or/automata_0.anml -i /home/tge/workspace/ngAP/automata_benchmark_original/AutomataZoo/APPRNG/benchmarks/4_sided/inputs/10MB_A.prng --app-name=APPRNG4 --algorithm=nonblockingallgroups --input-start-pos=0 --input-len=1000000 --split-entire-inputstream-to-chunk-size=1000000 --group-num=1 --duplicate-input-stream=600 --unique=false --unique-frequency=10 --use-soa=false --result-capacity=54619400 --use-uvm=false --data-buffer-fetch-size=512 --add-aan-start=1024 --add-aas-interval=4096 --active-threshold=8 --precompute-cutoff=-1 --precompute-depth=3 --compress-prec-table=true --pc-use-uvm=false --report-off=false --remove-degree=false --quit-degree=false --max-nfa-size=-1 --quick-validation=500000000 --validation=true 


### Latency
CAV: 0.163623 -> 1.444538
asyncap -a /home/tge/workspace/ngAP/automata_benchmark_original/AutomataZoo/ClamAV/benchmarks/anml_remove_or/automata_0.anml -i /home/tge/workspace/ngAP/automata_benchmark_original/AutomataZoo/ClamAV/benchmarks/inputs/clamav.input --app-name=ClamAV --algorithm=runahead --input-len=1000000 --report-off=true --duplicate-input-stream=1 --one-output-capacity=10461940 --scanning-R=999999999 --block-size=128 --record-ir=0 --blockDimX=-1 --num-streams=4 --merge-cc=4 --shrmem-wl=1 --shr_wl_len=4 --remove-degree=false --quit-degree=false --quick-validation=1 --validation=false 

CAV': 0.166491 -> 1.492108
asyncap -a /home/tge/workspace/ngAP/automata_benchmark_original/AutomataZoo/ClamAV/benchmarks/anml_remove_or/automata_0.anml -i /home/tge/workspace/ngAP/automata_benchmark_original/AutomataZoo/ClamAV/benchmarks/inputs/clamav.input --app-name=smallClamAV --algorithm=runahead --input-len=1000000 --report-off=true --duplicate-input-stream=1 --one-output-capacity=10461940 --scanning-R=999999999 --block-size=128 --record-ir=0 --blockDimX=-1 --num-streams=4 --merge-cc=4 --shrmem-wl=1 --shr_wl_len=4 --remove-degree=true --quit-degree=false --quick-validation=1 --validation=false 