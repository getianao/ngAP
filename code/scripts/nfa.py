import xml.etree.ElementTree as ET
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
import networkx as nx
import re
import os
import sys
from bitarray import bitarray

class State:

    def __init__(
        self, name="", id=-1, symbol_set="", start=-1, report=-1, loop_body=None
    ):
        self.name = name
        self.id = id
        self.symbol_set_expr = symbol_set
        self.start = start  # 0="start-of-data", 1="all-input"
        self.report = report # 1="report"
        self.report_code = None
        self.neighbors = []
        self.previous = []
        self.insts = []
        self.marker_stream = None
        self.result_stream = None

    def is_report(self):
        return self.report > 0

    def match(self, symbol):
        return bool(re.fullmatch(self.symbol_set_expr, symbol))

    def __str__(self):
        return f"State(name={self.name}, id={self.id}, symbol_set_expr={self.symbol_set_expr}, start={self.start}, report={self.report})"

    def __repr__(self):
        return f"State(name={self.name}, id={self.id}, symbol_set={self.symbol_set_expr}, start={self.start}, report={self.report})"


class NFA:
    def __init__(self, anml_path = None):
        self.anml_path = anml_path
        self.states = {}
        self.state_name_map = None
        self.state_num = 0
        self.edge_num = 0

        self.graph = None
        self.graph_coo = None
        self.graph_csr = None
        self.symbol_set_map = {}
        self.start_active_map = {}
        self.always_active_map = {}
        self.report_map = {}
        
        
    def from_states(self, states):
        self.states = states
        self.state_num = len(states)
        shape = (self.state_num, self.state_num)
        row = []
        col = []
        self.state_name_map = {}
        for state in states:
            self.state_name_map[state.name] = state.id
        for state in states:
            state_name = state.name
            if state_name not in self.state_name_map:
                raise ValueError("State name not found in state_name_map")
            state_id = self.state_name_map[state_name]
            symbol_set = state.symbol_set_expr
            self.symbol_set_map[state_id] = symbol_set
            start_int = state.start
            if start_int == 0:
                self.start_active_map[state_id] = 1
            elif start_int == 1:
                self.always_active_map[state_id] = 1
            report = state.report
            if report > 0:
                self.report_map[state_id] = 1
            neighbors = state.neighbors
            for neighbor in neighbors:
                row.append(state_id)
                col.append(neighbor)
                if neighbor > self.state_num:
                    raise ValueError(f"Neighbor {neighbor} out of bounds for state_num {self.state_num}")
        data = np.ones(len(row), dtype=int)
        self.edge_num = len(row)
        self.graph_coo = coo_matrix((data, (row, col)), shape=shape)
        self.graph_csr = self.graph_coo.tocsr()
        self.graph = nx.from_scipy_sparse_array(self.graph_csr, create_using=nx.DiGraph)
        
    def to_anml(self, output_anml_path):
        with open(output_anml_path, "w") as f:
            f.write('<anml version="1.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">\n')
            f.write('  <automata-network  id="tge">\n')
            for state in self.states:
                symbol_set = state.symbol_set_expr
                if state.start == 0:
                    start_string = "start-of-data"
                elif state.start == 1:
                    start_string = "all-input"
                else:
                    start_string = "none"
                f.write(f'    <state-transition-element id="{state.name}" symbol-set="{symbol_set}" start="{start_string}">\n')
                if state.is_report():
                    f.write(f'      <report-on-match reportcode="{state.report_code}"/>\n')
                for neighbor in state.neighbors:
                    f.write(f'      <activate-on-match element="{self.states[neighbor].name}"/>\n')
                f.write("    </state-transition-element>\n")
            f.write("  </automata-network>\n")
            f.write("</anml>\n")
            
        

    def get_state_id(self, state_name):
        if self.state_name_map == None:
            raise ValueError("Please load ANML file first.")
        if state_name not in self.state_name_map:
            self.state_name_map[state_name] = len(self.state_name_map)
        return self.state_name_map[state_name]

    def start_string_to_int(self, start_string):
        if start_string == "start-of-data":
            return 0
        elif start_string == "all-input":
            return 1
        elif start_string == "none" or start_string == None:
            return -1
        else:
            raise ValueError("Unknown start string: ", start_string)

    def load_anml(self):
        tree = ET.parse(self.anml_path)
        network = tree.getroot().find("automata-network")
        self.state_num = len(list(network))
        shape = (self.state_num, self.state_num)
        row = []  # from
        col = []  # to
        self.state_name_map = {}
        for state in network.iter("state-transition-element"):
            state_name = state.get("id")
            state_id = self.get_state_id(state_name)
            symbol_set = state.get("symbol-set")
            self.symbol_set_map[state_id] = symbol_set
            start_string = state.get("start")  # "start-of-data", "all-input"
            start_int = self.start_string_to_int(start_string)
            if start_int == 0:
                self.start_active_map[state_id] = 1
            elif start_int == 1:
                self.always_active_map[state_id] = 1
            reports = state.findall("report-on-match")
            assert len(reports) <= 1
            if len(reports) == 1:
                report_code = reports[0].get("reportcode")
                self.report_map[state_id] = 1
            neighbors = state.findall("activate-on-match")
            state = State(state_name, state_id, symbol_set, start_int, len(reports))
            if len(reports):
                state.report_code = report_code
            self.states[state_id] = state
            for neighbor in neighbors:
                neighbor_name = neighbor.get("element")
                row.append(state_id)
                col.append(self.get_state_id(neighbor_name))
                state.neighbors.append(self.get_state_id(neighbor_name))
            
        data = np.ones(len(row), dtype=int)
        self.edge_num = len(row)
        self.graph_coo = coo_matrix((data, (row, col)), shape=shape)
        self.graph_csr = self.graph_coo.tocsr()
        self.graph = nx.from_scipy_sparse_array(self.graph_csr, create_using=nx.DiGraph)

    # Translate from Jack Wadden's VASim by chatgpt
    def symbol_set_to_bitarray(self, symbol_set: str):

        def set_range(b: bitarray, start: int, end: int, value: int):
            for i in range(start, end + 1):
                b[i] = value

        column = bitarray(256)
        if symbol_set == "*":
            column.setall(1)
            return

        if symbol_set == ".":
            column.setall(1)
            column[ord("\n")] = 0
            return

        in_charset = False
        escaped = False
        inverting = False
        range_set = False
        bracket_sem = 0
        brace_sem = 0
        value = 1
        last_char = 0
        range_start = 0

        OPEN_BRACKET = 256

        if symbol_set.startswith("{") and symbol_set.endswith("}"):
            print("CURLY BRACES NOT IMPLEMENTED")
            exit(1)

        index = 0
        while index < len(symbol_set):
            c = symbol_set[index]

            if c == "[":
                if escaped:
                    column[ord(c)] = value
                    if range_set:
                        set_range(column, range_start, ord(c), value)
                        range_set = False
                    last_char = c
                    escaped = False
                else:
                    last_char = OPEN_BRACKET
                    bracket_sem += 1
            elif c == "]":
                if escaped:
                    column[ord(c)] = value
                    if range_set:
                        set_range(column, range_start, ord(c), value)
                        range_set = False
                    escaped = False
                    last_char = c
                else:
                    bracket_sem -= 1
            elif c == "\\":
                if escaped:
                    column[ord(c)] = value
                    if range_set:
                        set_range(column, range_start, ord(c), value)
                        range_set = False
                    last_char = c
                    escaped = False
                else:
                    escaped = True
            elif c in {"n", "r", "t", "a", "b", "f", "v", "'", '"'}:
                char_map = {
                    "n": "\n",
                    "r": "\r",
                    "t": "\t",
                    "a": "\a",
                    "b": "\b",
                    "f": "\f",
                    "v": "\v",
                    "'": "'",
                    '"': '"',
                }
                actual_char = ord(char_map[c]) if escaped else ord(c)
                column[actual_char] = value
                if range_set:
                    set_range(column, range_start, actual_char, value)
                    range_set = False
                last_char = actual_char
                escaped = False
            elif c == "-":
                if escaped or last_char == OPEN_BRACKET:
                    column[ord("-")] = value
                    if range_set:
                        set_range(column, range_start, ord("-"), value)
                        range_set = False
                    escaped = False
                    last_char = "-"
                else:
                    range_set = True
                    range_start = last_char
            elif c == "s" and escaped:
                special_chars = ["\n", "\t", "\r", "\x0B", "\x0C", " "]
                for char in special_chars:
                    column[ord(char)] = value
                escaped = False
            elif c == "d" and escaped:
                set_range(column, ord("0"), ord("9"), value)
                escaped = False
            elif c == "w" and escaped:
                column[ord("_")] = value
                set_range(column, ord("0"), ord("9"), value)
                set_range(column, ord("A"), ord("Z"), value)
                set_range(column, ord("a"), ord("z"), value)
                escaped = False
            elif c == "^" and not escaped:
                inverting = True
            elif c == "x" and escaped:
                hex_char = symbol_set[index + 1 : index + 3]
                number = int(hex_char, 16)
                column[number] = value
                if range_set:
                    set_range(column, range_start, number, value)
                    range_set = False
                last_char = number
                index += 2
                escaped = False
            else:
                if escaped:
                    escaped = False
                column[ord(c)] = value
                if range_set:
                    set_range(column, range_start, ord(c), value)
                    range_set = False
                last_char = ord(c)
            index += 1

        if inverting:
            column.invert()

        if bracket_sem != 0 or brace_sem != 0:
            print(f"MALFORMED BRACKETS OR BRACES: {symbol_set}")
            print(f"brackets: {bracket_sem}")
            exit(1)

        return column

    def stats(self):
        print("State number: ", self.state_num)
        print("always-active state number: ", len(self.always_active_map))
        print("start state number: ", len(self.start_active_map))
        print("report state number: ", len(self.report_map))
        print("Edge number: ", self.edge_num)
        connected_components = list(nx.weakly_connected_components(self.graph))
        num_connected_components = len(connected_components)
        print("CC number: ", num_connected_components)
        print(" max CC size: ", max([len(cc) for cc in connected_components]))
        print(" avg CC size: ", np.mean([len(cc) for cc in connected_components]))

        if isinstance(self.graph, nx.DiGraph):
            in_degrees = [deg for node, deg in self.graph.in_degree()]
            out_degrees = [deg for node, deg in self.graph.out_degree()]
            avg_in_degree = sum(in_degrees) / self.state_num
            avg_out_degree = sum(out_degrees) / self.state_num
            max_in_degree = max(in_degrees)
            max_out_degree = max(out_degrees)
            print("Average in-degree: ", avg_in_degree)
            print("Average out-degree: ", avg_out_degree)
            print("Max in-degree: ", max_in_degree)
            print("Max out-degree: ", max_out_degree)

            # Loop
            print("Loop number: ", len(list(nx.simple_cycles(self.graph))))
            print("self loop number: ", len(list(nx.nodes_with_selfloops(self.graph))))

            loop_symbol_set_type = {}

            selected_self_loop_num = 0
            select_condition = 254
            selected_degree = 0
            selected_degree_2 = 0
            for node in nx.nodes_with_selfloops(self.graph):
                symbol_set = self.symbol_set_map[node]
                if symbol_set in loop_symbol_set_type:
                    loop_symbol_set_type[symbol_set] += 1
                else:
                    loop_symbol_set_type[symbol_set] = 1
                symbol_set_bitarray = self.symbol_set_to_bitarray(symbol_set)
                if symbol_set_bitarray.count() > select_condition:
                    selected_self_loop_num += 1
                    selected_degree += self.graph.out_degree(node)
                    if self.graph.out_degree(node) == 2:
                        selected_degree_2 += 1
                # print(
                #     node,
                #     self.always_active_map[node] if node in self.always_active_map else 0,
                #     self.start_active_map[node] if node in self.start_active_map else 0,
                #     self.report_map[node] if node in self.report_map else 0,
                #     file=sys.stderr,
                # )
                # print(self.symbol_set_map[node], file=sys.stderr)
            loop_symbol_set_type = dict(
                sorted(loop_symbol_set_type.items(), key=lambda item: item[1], reverse=True)
            )
            print(f"Selected self loop number (bit num > {select_condition}): {selected_self_loop_num}")
            print(
                "Loop symbol set type length: ",
                len(loop_symbol_set_type),
            )
            if selected_self_loop_num > 0:
                print(
                    "Selected self loop degree: ",
                    selected_degree / selected_self_loop_num,
                )
            print(f"Selected self loop degree 2: {selected_degree_2}")

            # print(
            #     "Loop symbol set type: ",
            #     loop_symbol_set_type,
            #     file=sys.stderr,
            # )
            symbol_set_type = {}
            for node in self.graph.nodes():
                symbol_set = self.symbol_set_map[node]
                if symbol_set in symbol_set_type:
                    symbol_set_type[symbol_set] += 1
                else:
                    symbol_set_type[symbol_set] = 1
            print(f"symbol_set_type length: {len(symbol_set_type)}")
            print(f"symbol_set_type sum: {sum(symbol_set_type.values())}")
            # print("symbol_set_type: ", symbol_set_type, file=sys.stderr)

            # for node in self.graph.nodes():
            #     if self.graph.out_degree(node) == 0:
            #         if node not in self.report_map:
            #             print("report_map??? ", node, self.symbol_set_map[node], file=sys.stderr)

            successors_set_dict = {}
            sequence_node_num = 0
            for node in self.graph.nodes():
                successors_list = list(self.graph.successors(node))
                successors = self.graph.successors(node)
                successors_set  = frozenset(successors)
                ss = (self.symbol_set_map[node], successors_set)
                if ss in successors_set_dict:
                    successors_set_dict[ss] += 1
                else:
                    successors_set_dict[ss] = 1
                # print(successors_list, file=sys.stderr)

                if ((node + 1) in successors_list):
                    sequence_node_num += 1

            print(f"successors_set_dict_length: {len(successors_set_dict)}")
            print(f"successors_set_dict_percent: {len(successors_set_dict) / len(self.graph.nodes())}")
            print(f"sequence_node_num: {sequence_node_num}")
            print(f"sequence_node_percent: {sequence_node_num / len(self.graph.nodes())}")
            print(f"sequence_node_percent: {sequence_node_num / len(self.graph.nodes())}", file=sys.stderr)    

    def get_always_active_states(self):
        states = []
        for state_id in self.always_active_map.keys():
            states.append(self.states[state_id])
        return states

    def get_start_states(self):
        states = []
        for state_id in self.start_active_map.keys():
            states.append(self.states[state_id])
        return states

    def get_neighbor_states(self, state):
        states = []
        for neighbor_id in self.graph_csr[state.id].indices:
            states.append(self.states[neighbor_id])
        return states

    def init_neighbor_states(self):
        for state in self.states.values():
            state.neighbors = self.get_neighbor_states(state)

    def parse(self):
        self.load_anml()
        self.init_neighbor_states()

    def visualize(self):
        import matplotlib.pyplot as plt
        if (self.state_num > 32):
            print("NFA state number > 32. Ignore NFA visualization.")
            return

        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos, with_labels=True)
        plt.savefig("graph.png")

    def check_loop_overlapping(self):
        from itertools import combinations
        cycles = list(nx.simple_cycles(self.graph))
        for (cycle1, cycle2) in combinations(cycles, 2):
            if set(cycle1) & set(cycle2):
                print("Found overlapping loop")
                print("cycle1", cycle1)
                print("cycle2", cycle2)
                # exit(0)
                return True, cycle1, cycle2  # 返回重叠的环
        print("Not Found overlapping loop")
        return False, None, None


if __name__ == "__main__":
    nfa = NFA(
        "/home/tge/workspace/ngap2/dataset/AutomataZoo/Brill/benchmarks/anml_remove_or/automata_0.anml"
    )
    nfa.load_anml()
    nfa.stats()
