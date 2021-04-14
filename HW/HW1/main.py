from typing import List, Set, Dict
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import networkx
import copy


def LTM(graph: networkx.Graph, patients_0: List, iterations: int) -> Set:

    total_infected = set(patients_0)
    susceptible = set(graph.nodes) - total_infected
    removed = set()


    for i in range(iterations):

        I_t_minus_1 = total_infected
        I_t = set()

        targets = propagate(graph, I_t_minus_1)

        pass


    return total_infected




def ICM(graph: networkx.Graph, patients_0: List, iterations: int) -> [Set, Set]:
    total_infected = set(patients_0)
    total_deceased = set()
    # TODO implement your code here
    return total_infected, total_deceased


def propagate(graph, NI):
    targets = []
    for node in NI:
        targets += graph.neighbors(node)
    return targets

def propagate_dict(graph, NI):
    targets = {}
    for node in NI:
        for neighbor in graph.neighbors(node):
            if neighbor not in targets:
                targets[neighbor] = [node]
            else:
                targets[neighbor].append(node)
    return targets


def plot_degree_histogram(histogram: Dict):
    plt.bar(list(histogram.keys()), histogram.values(), color='g')
    plt.show()
    return


def calc_degree_histogram(graph: networkx.Graph) -> Dict:
    """
    Example:
    if histogram[1] = 10 -> 10 nodes have only 1 friend
    """
    histogram = {}
    for i in graph.degree():
        degree = i[1]
        if not degree in histogram:
            histogram[degree] = 1
        else:
            histogram[degree] += 1
    return histogram


def build_graph(filename: str) -> networkx.Graph:
    Graphtype = networkx.Graph()
    df = pd.read_csv(filename)
    if filename == 'PartB-C.csv':
        G = networkx.from_pandas_edgelist(df, source="from", target="to",
                                          edge_attr=['w'], create_using=Graphtype)
    else:
        G = networkx.from_pandas_edgelist(df, source="from", target="to", create_using=Graphtype)
    return G


def clustering_coefficient(graph: networkx.Graph) -> float:
    numerator = sum(networkx.triangles(graph).values())

    denominator = 0
    histogram = calc_degree_histogram(graph)
    for degree in histogram:
        # a node with degree d is part of d choose 2 triplets
        denominator += degree * (degree - 1) / 2 * histogram[degree]

    cc = numerator / denominator
    return cc


def compute_lethality_effect(graph: networkx.Graph, t: int) -> [Dict, Dict]:
    global LETHALITY
    mean_deaths = {}
    mean_infected = {}
    for l in (.05, .15, .3, .5, .7):
        LETHALITY = l
        for iteration in range(30):
            G = copy.deepcopy(graph)
            patients_0 = np.random.choice(list(G.nodes), size=50, replace=False, p=None)
            # TODO implement your code here

    return mean_deaths, mean_infected


def plot_lethality_effect(mean_deaths: Dict, mean_infected: Dict):
    # TODO implement your code here
    ...


def choose_who_to_vaccinate(graph: networkx.Graph) -> List:
    people_to_vaccinate = []
    # TODO implement your code here
    return people_to_vaccinate


def choose_who_to_vaccinate_example(graph: networkx.Graph) -> List:
    """
    The following heuristic for Part C is simply taking the top 50 friendly people;
     that is, it returns the top 50 nodes in the graph with the highest degree.
    """
    node2degree = dict(graph.degree)
    sorted_nodes = sorted(node2degree.items(), key=lambda item: item[1], reverse=True)[:50]
    people_to_vaccinate = [node[0] for node in sorted_nodes]
    return people_to_vaccinate


"Global Hyper-parameters"
CONTAGION = 1
LETHALITY = .15

if __name__ == "__main__":
    filename_1 = "PartA1.csv"
    filename_2 = "PartA2.csv"
    filename_3 = "PartB-C.csv"


    # G1 = build_graph(filename=filename_1)
    # G2 = build_graph(filename=filename_2)
    #
    # histogram_1 = calc_degree_histogram(G1)
    # histogram_2 = calc_degree_histogram(G2)
    # plot_degree_histogram(histogram_1)
    # plot_degree_histogram(histogram_2)
    #
    # print(clustering_coefficient(G1))
    # print(clustering_coefficient(G2))


    G3 = build_graph(filename_3)
    LTM(G3, [0,1,2,3], 10)
