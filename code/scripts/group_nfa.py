from nfa import NFA, State
import networkx as nx
import copy
import random
import os
import sys


def group_nfa(nfa_instance, nfa_number):
    ccs = nx.weakly_connected_components(nfa_instance.graph)
    ccs = sorted(ccs, key=len, reverse=True)
    grouped_ccs = []
    for i in range(nfa_number):
        grouped_ccs.append([])
    for idx, cc in enumerate(ccs):
        min_group = min(grouped_ccs, key=lambda group: sum(len(cc) for cc in group))
        min_group.append(list(cc))

    nfas = []
    for ccs in grouped_ccs:
        # print(ccs)
        # print(f"Size of ccs={len(ccs)}")
        # print(f"Size of node={sum([len(cc) for cc in ccs])}")
        nfa_states = []
        state_id_mapping = {}
        state_id_sum = 0
        for cc in ccs:
            for node_id, node in enumerate(cc):
                if node not in state_id_mapping:
                    state_id_mapping[node] = state_id_sum
                    state_id_sum += 1
                else:
                    raise ValueError(f"Node {node} already exists in state_id_mapping")
                
        # Update neighbors and state with new state ids
        for cc in ccs:
            for node_id, node in enumerate(cc):
                state = copy.copy(nfa_instance.states[node])
                new_state_id = state_id_mapping[node]
                state.id = new_state_id
                new_neighbors = []
                for neighbor in state.neighbors:
                    if neighbor in state_id_mapping:
                        new_neighbors.append(state_id_mapping[neighbor])
                    else:
                        raise ValueError(
                            f"Neighbor {neighbor} not found in state_id_mapping"
                        )
                state.neighbors = new_neighbors
                nfa_states.append(state)
        nfa = NFA()
        # print(f"Size of nfa_states={len(nfa_states)}")
        nfa.from_states(nfa_states)
        nfas.append(nfa)
    return nfas

def group_anml(anml_path, output_path, nfa_number):
    skip  = False
    group_anmls = [] 
    if skip:
        for i in range(nfa_number):
            new_anml_path = os.path.join(
                output_path, f"{os.path.basename(anml_path)}_grouped_{i}.anml"
            )
            if not os.path.exists(new_anml_path):
                raise ValueError(
                    f"File {new_anml_path} does not exist, please check the path"
                )
            group_anmls.append(new_anml_path)
        return group_anmls
    nfa = NFA(anml_path)
    nfa.load_anml()
    nfas = group_nfa(nfa, nfa_number)
    base_name = os.path.basename(anml_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    for new_nfa_idx, new_nfa in enumerate(nfas):
        new_anml_path = os.path.join(
            output_path, f"{base_name}_grouped_{new_nfa_idx}.anml"
        )
        if skip and os.path.exists(new_anml_path):
            group_anmls.append(new_anml_path)
            continue
        new_nfa.to_anml(new_anml_path)
        group_anmls.append(new_anml_path)
        print(f"Saved new NFA to {new_anml_path}")
    return group_anmls  

if __name__ == "__main__":
    group_anml(
        "/home/tge/workspace/ngap2/dataset/AutomataZoo/YARA/benchmarks/YARA/anml_remove_or/automata_0.anml",
        "/home/tge/workspace/ngap2/dataset_group_nfa/12/YARA/",
        12,
    )
