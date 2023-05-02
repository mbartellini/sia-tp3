import os.path
import tracemalloc
import time
from abc import ABC
from numbers import Number
from typing import List, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.algorithms import BfsAlgorithm, DfsAlgorithm, GreedyAlgorithm, AStarAlgorithm, Algorithm
from src.fill_zone.heuristics import EccentricityHeuristic, ColorCountHeuristic, CombinationHeuristic, \
    NodeCountHeuristic
from src.fill_zone.state import FillZoneGraphState
from src.heuristics import DummyHeuristic, Heuristic
from src.result import Result
from src.search_tree import SearchTree
from src.state import State

OUTPUT_DIR = "figs/"
M = 5
N = 5
TEST_COUNT = 100
UNINFORMED_ALGOS = [
    BfsAlgorithm(),
    DfsAlgorithm()
]
HEURISTICS = [
    EccentricityHeuristic(),
    ColorCountHeuristic()
]
INFORMED_ALGOS = [
    GreedyAlgorithm(),
    AStarAlgorithm()
]

OPTIMAL_PAIRS = [
    (BfsAlgorithm(), DummyHeuristic()),
    (AStarAlgorithm(), EccentricityHeuristic()),
    (AStarAlgorithm(), ColorCountHeuristic()),
    (AStarAlgorithm(), CombinationHeuristic())
]
NOT_OPTIMAL_PAIRS = [
    (DfsAlgorithm(), DummyHeuristic()),
    (AStarAlgorithm(), NodeCountHeuristic()),
    (GreedyAlgorithm(), EccentricityHeuristic()),
    (GreedyAlgorithm(), ColorCountHeuristic()),
    (GreedyAlgorithm(), CombinationHeuristic()),
    (GreedyAlgorithm(), NodeCountHeuristic()),

]
MEMORY_ANAL = [
    (BfsAlgorithm(), DummyHeuristic()),
    (DfsAlgorithm(), DummyHeuristic()),
    (AStarAlgorithm(), CombinationHeuristic())
]


def _get_alias(o: object):
    return o.__class__.__name__[:3]


def _get_key(algoheu: Tuple[Algorithm, Heuristic]) -> str:
    return f"{_get_alias(algoheu[0])}+{_get_alias(algoheu[1])}"


class PlotSupplier(ABC):

    def __init__(self, out_name: str):
        self._out_name = out_name

    def plot(self):
        data = self._create_data()

        for t in range(TEST_COUNT):
            self._add_data(data, self._get_new_data())

        data = self._post_process(data)

        self._save_plot(data)

    def _create_data(self):
        raise NotImplementedError()

    def _get_new_data(self):
        raise NotImplementedError()

    def _add_data(self, data, new_data):
        raise NotImplementedError

    def _post_process(self, data):
        raise NotImplementedError()

    def _save_plot(self, data):
        raise NotImplementedError()


class BarPlotSupplier(PlotSupplier, ABC):
    def __init__(self, out_name: str, yaxis_name: str, algoheus: List[Tuple[Algorithm, Heuristic]]):
        super().__init__(out_name)
        self._algoheus = algoheus
        self._yaxis_name = yaxis_name

    def _create_data(self):
        data: Dict[str, List[Number]] = {}
        for algoheu in self._algoheus:
            data[_get_key(algoheu)] = []
        return data

    def _get_new_data(self) -> Dict[str, Number]:
        new_data: Dict[str, Number] = {}
        b: List[List[int]] = np.random.randint(0, M, (N, N)).tolist()
        for algoheu in self._algoheus:
            g: State = FillZoneGraphState(b)
            search_tree = SearchTree(g, algoheu[1])
            new_data[_get_key(algoheu)] = self._get_useful_data(search_tree, algoheu[0])
        return new_data

    def _get_useful_data(self, search_tree: SearchTree, a: Algorithm) -> Number:
        raise NotImplementedError()

    def _add_data(self, data, new_data):
        for key in new_data.keys():
            data[key].append(new_data[key])

    def _post_process(self, data):
        algos = list(data.keys())
        return [
            algos,
            np.arange(len(algos)),
            [np.mean(data[algo]) for algo in algos],
            [np.std(data[algo]) for algo in algos]
        ]

    def _save_plot(self, data):
        algos = data[0]
        x_pos = data[1]
        means = data[2]
        stds = data[3]

        fig, ax = plt.subplots()
        ax.bar(x_pos, means, yerr=stds, align='center', alpha=0.5, ecolor='black')
        ax.set_ylabel(self._yaxis_name)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(algos, rotation='vertical')
        ax.set_title('Algorithm')
        ax.yaxis.grid(True)
        ax.figure.autofmt_xdate()
        plt.savefig(OUTPUT_DIR + self._out_name + "_plot")
        plt.show()


class OptimalTimeBarPlotSupplier(BarPlotSupplier):
    def __init__(self):
        super().__init__("optimal_times", "Time in s", OPTIMAL_PAIRS)

    def _get_useful_data(self, search_tree: SearchTree, a: Algorithm) -> Number:
        start_time = time.time()
        search_tree.search(a)
        end_time = time.time()
        return end_time - start_time


class OptimalExpandedBarPlotSupplier(BarPlotSupplier):
    def __init__(self):
        super().__init__("optimal_expanded_nodes", "Expanded nodes", OPTIMAL_PAIRS)

    def _get_useful_data(self, search_tree: SearchTree, a: Algorithm) -> Number:
        res: Result = search_tree.search(a)
        return res.expanded_nodes


class OptimalFrontierBarPlotSupplier(BarPlotSupplier):
    def __init__(self):
        super().__init__("optimal_frontier_nodes", "Frontier nodes", OPTIMAL_PAIRS)

    def _get_useful_data(self, search_tree: SearchTree, a: Algorithm) -> Number:
        res: Result = search_tree.search(a)
        return res.frontier_nodes


class NotOptimalTimeBarPlotSupplier(BarPlotSupplier):
    def __init__(self):
        super().__init__("not_optimal_times", "Time in s", NOT_OPTIMAL_PAIRS)

    def _get_useful_data(self, search_tree: SearchTree, a: Algorithm) -> Number:
        start_time = time.time()
        search_tree.search(a)
        end_time = time.time()
        return end_time - start_time


class NotOptimalExpandedBarPlotSupplier(BarPlotSupplier):
    def __init__(self):
        super().__init__("not_optimal_expanded_nodes", "Expanded nodes", NOT_OPTIMAL_PAIRS)

    def _get_useful_data(self, search_tree: SearchTree, a: Algorithm) -> Number:
        res: Result = search_tree.search(a)
        return res.expanded_nodes


class NotOptimalFrontierBarPlotSupplier(BarPlotSupplier):
    def __init__(self):
        super().__init__("not_optimal_frontier_nodes", "Frontier nodes", NOT_OPTIMAL_PAIRS)

    def _get_useful_data(self, search_tree: SearchTree, a: Algorithm) -> Number:
        res: Result = search_tree.search(a)
        return res.frontier_nodes


class MemoryBarPlotSupplier(BarPlotSupplier):
    def __init__(self):
        super().__init__("memory", "Memory Usage", MEMORY_ANAL)

    def _get_useful_data(self, search_tree: SearchTree, a: Algorithm) -> Number:
        tracemalloc.start()
        res: Result = search_tree.search(a)
        value = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return value


# tiempos de optimos
# expanded nodes optimos
# frontier optimos
# tiempos de no optimos
# expanded nodes no optimos
# frontier no optimos
# memory

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    tests = [
        OptimalTimeBarPlotSupplier(),
        OptimalExpandedBarPlotSupplier(),
        OptimalFrontierBarPlotSupplier(),
        NotOptimalTimeBarPlotSupplier(),
        NotOptimalExpandedBarPlotSupplier(),
        NotOptimalFrontierBarPlotSupplier(),
        MemoryBarPlotSupplier()
    ]
    for test in tests:
        test.plot()