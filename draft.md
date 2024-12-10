# fermi OA-default

ngap -a /home/tge/workspace/ngAP/automata_benchmark_original/ANMLZoo/Fermi/anml_remove_or/automata_0.anml -i /home/tge/workspace/ngAP/automata_benchmark_original/ANMLZoo/Fermi/inputs/rp_input_10MB.input --app-name=Fermi  --algorithm=nonblockingallgroups  --input-start-pos=0  --input-len=1000000  --split-entire-inputstream-to-chunk-size=1000000 --group-num=1  --duplicate-input-stream=1  --unique=true  --unique-frequency=10  --use-soa=false  --result-capacity=54619400  --use-uvm=false  --data-buffer-fetch-size=25600  --add-aan-start=256  --add-aas-interval=1000  --active-threshold=0  --precompute-cutoff=-1  --precompute-depth=2  --compress-prec-table=true  --pc-use-uvm=false  --report-off=false  --remove-degree=false  --quit-degree=false  --max-nfa-size=-1  --adaptive-aas=false --quick-validation=-1 --validation=true

# fermi O1

ngap -a /home/tge/workspace/ngAP/automata_benchmark_original/ANMLZoo/Fermi/anml_remove_or/automata_0.anml -i /home/tge/workspace/ngAP/automata_benchmark_original/ANMLZoo/Fermi/inputs/rp_input_10MB.input --app-name=Fermi  --algorithm=nonblockinggroups  --input-start-pos=0  --input-len=1000  --split-entire-inputstream-to-chunk-size=1000 --group-num=1  --duplicate-input-stream=1  --unique=true  --unique-frequency=10  --use-soa=false  --result-capacity=54619400  --use-uvm=false  --data-buffer-fetch-size=25600  --add-aan-start=256  --add-aas-interval=1000  --active-threshold=0  --precompute-cutoff=-1  --precompute-depth=2  --compress-prec-table=true  --pc-use-uvm=false  --report-off=false  --remove-degree=false  --quit-degree=false  --max-nfa-size=-1  --adaptive-aas=false --quick-validation=-1 --validation=true


ngap -a /home/tge/workspace/ngAP/automata_benchmark_original/ANMLZoo/Fermi/anml_remove_or/automata_0.anml -i /home/tge/workspace/ngAP/automata_benchmark_original/ANMLZoo/Fermi/inputs/rp_input_10MB.input --app-name=Fermi  --algorithm=nonblockinggroups  --input-start-pos=5000000  --input-len=5000000  --split-entire-inputstream-to-chunk-size=5000000 --group-num=1  --duplicate-input-stream=1  --unique=false  --unique-frequency=10  --use-soa=false  --result-capacity=54619400  --use-uvm=false  --data-buffer-fetch-size=25600  --add-aan-start=256  --add-aas-interval=1000000  --active-threshold=0  --precompute-cutoff=-1  --precompute-depth=2  --compress-prec-table=true  --pc-use-uvm=false  --report-off=false  --remove-degree=false  --quit-degree=false  --max-nfa-size=-1  --adaptive-aas=false --quick-validation=-1 --validation=true



## Poweren

ngap -a /home/tge/workspace/ngAP/automata_benchmark_original/ANMLZoo/PowerEN/anml_remove_or/automata_0.anml -i /home/tge/workspace/ngAP/automata_benchmark_original/ANMLZoo/PowerEN/inputs/poweren_10MB.input --app-name=PowerEN  --algorithm=nonblockinggroups  --input-start-pos=0  --input-len=1000000  --split-entire-inputstream-to-chunk-size=1000000  --group-num=1  --duplicate-input-stream=1  --unique=false  --unique-frequency=10  --use-soa=false  --result-capacity=54619400  --use-uvm=false  --data-buffer-fetch-size=25600  --add-aan-start=256  --add-aas-interval=64  --active-threshold=0  --precompute-cutoff=-1  --precompute-depth=3  --compress-prec-table=true  --pc-use-uvm=false  --report-off=false  --remove-degree=false  --quit-degree=false  --max-nfa-size=-1  --adaptive-aas=true --quick-validation=4304 --validation=true


<state-transition-element id="__9046__"  symbol-set="[\x00-\x21\x23-\xff]"  start="none">
	<activate-on-match element="__9046__"/>
	<activate-on-match element="__9047__"/>
</state-transition-element>

symbol-set="\[\\x00-\\x09\\x0b-\\xff\]"  start="none">
	<activate-on-match element.*


# Compare uniqe
## snort

ngap -a /home/tge/workspace/ngAP/automata_benchmark_original/AutomataZoo/Snort/benchmarks/anml_remove_or/automata_0.anml -i /home/tge/workspace/ngAP/automata_benchmark_original/AutomataZoo/Snort/benchmarks/inputs/wrccdc2012.pcap --app-name=Snort  --algorithm=nonblockingallgroups  --input-start-pos=0  --input-len=1000000  --split-entire-inputstream-to-chunk-size=1000000  --group-num=1  --duplicate-input-stream=600  --unique=false  --unique-frequency=10  --use-soa=false  --result-capacity=54619400  --use-uvm=false  --data-buffer-fetch-size=25600  --add-aan-start=256  --add-aas-interval=1000000  --active-threshold=0  --precompute-cutoff=-1  --precompute-depth=3  --compress-prec-table=true  --pc-use-uvm=false  --report-off=false  --remove-degree=false  --quit-degree=false  --max-nfa-size=-1  --adaptive-aas=true --quick-validation=128259 --validation=true

unique=1 : 63.4093 MB/s
unique=0: wrong

## Brill

ngap -a /home/tge/workspace/ngAP/automata_benchmark_original/AutomataZoo/Brill/benchmarks/anml_remove_or/automata_0.anml -i /home/tge/workspace/ngAP/automata_benchmark_original/AutomataZoo/Brill/benchmarks/inputs/brown_corpus.txt --app-name=Brill  --algorithm=nonblockingallgroups  --input-start-pos=0  --input-len=1000000  --split-entire-inputstream-to-chunk-size=1000000  --group-num=1  --duplicate-input-stream=600  --unique=true  --unique-frequency=128  --use-soa=false  --result-capacity=54619400  --use-uvm=false  --data-buffer-fetch-size=25600  --add-aan-start=256  --add-aas-interval=1000000  --active-threshold=0  --precompute-cutoff=-1  --precompute-depth=3  --compress-prec-table=true  --pc-use-uvm=false  --report-off=false  --remove-degree=false  --quit-degree=false  --max-nfa-size=-1  --adaptive-aas=true --quick-validation=9038877 --validation=true

unique=0 : 8.04247 MB/s
unique=1 : 7.94636


# Latency


256
82*1536/256 = 492 




## remove loop
time ${NGAP_ROOT}/scripts/run_throughput_gpu_nap_defalut_loop.sh



time ${NGAP_ROOT}/scripts/run_throughput_gpu_nap_defalut.sh


if (node_attrs[rvertex] & 0b100) {
      if (symbol_set.test(rvertex, rsymbol)) {
        addToBufferSimple(rvertex, riter + 1, d_buffer, d_buffer_idx,
                          *d_buffer_start, d_buffer_end_tmp,
                          buffer_capacity_per_block);
        if (node_attrs[rvertex] & 0b10)
          addResult2(rvertex, riter + 1, d_results_v, d_results_i, results_size,
                     nblb.results_capacity, nblb.report_off);
      }
    }

ngap -a /home/tge/workspace/ngAP/automata_benchmark_original/AutomataZoo/Brill/benchmarks/anml_remove_or/automata_0.anml -i /home/tge/workspace/ngAP/automata_benchmark_original/AutomataZoo/Brill/benchmarks/inputs/brown_corpus.txt --app-name=Brill  --algorithm=nonblockingallgroups  --input-start-pos=0  --input-len=1000000  --split-entire-inputstream-to-chunk-size=1000000  --group-num=1  --duplicate-input-stream=600  --unique=false  --unique-frequency=10  --use-soa=false  --result-capacity=54619400  --use-uvm=false  --data-buffer-fetch-size=25600  --add-aan-start=256  --add-aas-interval=1000000  --active-threshold=0  --precompute-cutoff=-1  --precompute-depth=3  --compress-prec-table=true  --pc-use-uvm=false  --report-off=false  --remove-degree=false  --quit-degree=false  --max-nfa-size=-1  --adaptive-aas=true --quick-validation=9038877 --validation=true



kernel_ngap_OAE1:  --use-unique-matchset=true --remove-loop-edge=false --loop-state-prefetch=false

ngap -a /home/tge/workspace/ngAP/automata_benchmark_original/AutomataZoo/Brill/benchmarks/anml_remove_or/automata_0.anml -i /home/tge/workspace/ngAP/automata_benchmark_original/AutomataZoo/Brill/benchmarks/inputs/brown_corpus.txt --app-name=Brill  --algorithm=nonblockingalle1groups  --input-start-pos=0  --input-len=1000000  --split-entire-inputstream-to-chunk-size=1000000  --group-num=1  --duplicate-input-stream=600  --unique=false  --unique-frequency=10  --use-soa=false  --result-capacity=54619400  --use-uvm=false  --data-buffer-fetch-size=25600  --add-aan-start=256  --add-aas-interval=1000000  --active-threshold=0  --precompute-cutoff=-1  --precompute-depth=3  --compress-prec-table=true  --pc-use-uvm=false  --report-off=false  --remove-degree=false  --quit-degree=false  --max-nfa-size=-1  --adaptive-aas=true --quick-validation=9038877 --validation=true --use-unique-matchset=true --remove-loop-edge=false --loop-state-prefetch=false

kernel_ngap_OAE2:  --use-unique-matchset=true --remove-loop-edge=true --loop-state-prefetch=false

ngap -a /home/tge/workspace/ngAP/automata_benchmark_original/AutomataZoo/Brill/benchmarks/anml_remove_or/automata_0.anml -i /home/tge/workspace/ngAP/automata_benchmark_original/AutomataZoo/Brill/benchmarks/inputs/brown_corpus.txt --app-name=Brill  --algorithm=nonblockingalle2groups  --input-start-pos=0  --input-len=1000000  --split-entire-inputstream-to-chunk-size=1000000  --group-num=1  --duplicate-input-stream=600  --unique=false  --unique-frequency=10  --use-soa=false  --result-capacity=54619400  --use-uvm=false  --data-buffer-fetch-size=25600  --add-aan-start=256  --add-aas-interval=1000000  --active-threshold=0  --precompute-cutoff=-1  --precompute-depth=3  --compress-prec-table=true  --pc-use-uvm=false  --report-off=false  --remove-degree=false  --quit-degree=false  --max-nfa-size=-1  --adaptive-aas=true --quick-validation=9038877 --validation=true --use-unique-matchset=true --remove-loop-edge=true --loop-state-prefetch=false

kernel_ngap_OAE2p:  --use-unique-matchset=true --remove-loop-edge=true --loop-state-prefetch=true

ngap -a /home/tge/workspace/ngAP/automata_benchmark_original/AutomataZoo/Brill/benchmarks/anml_remove_or/automata_0.anml -i /home/tge/workspace/ngAP/automata_benchmark_original/AutomataZoo/Brill/benchmarks/inputs/brown_corpus.txt --app-name=Brill  --algorithm=nonblockingalle2pgroups  --input-start-pos=0  --input-len=1000000  --split-entire-inputstream-to-chunk-size=1000000  --group-num=1  --duplicate-input-stream=600  --unique=false  --unique-frequency=10  --use-soa=false  --result-capacity=54619400  --use-uvm=false  --data-buffer-fetch-size=25600  --add-aan-start=256  --add-aas-interval=1000000  --active-threshold=0  --precompute-cutoff=-1  --precompute-depth=3  --compress-prec-table=true  --pc-use-uvm=false  --report-off=false  --remove-degree=false  --quit-degree=false  --max-nfa-size=-1  --adaptive-aas=true --quick-validation=9038877 --validation=true --use-unique-matchset=true --remove-loop-edge=true --loop-state-prefetch=true


time ${NGAP_ROOT}/scripts/run-throughput-extend.sh

${NGAP_ROOT}/scripts/gen-throughput-extend.sh

python ${NGAP_ROOT}/scripts/plot_throughput_gpu_sota_extend.py
python ${NGAP_ROOT}/scripts/plot_motivation_e1.py
python ${NGAP_ROOT}/scripts/plot_motivation_e2.py

