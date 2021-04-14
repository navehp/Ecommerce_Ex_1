from typing import List, Set, Dict
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import networkx
import copy


def LTM(graph: networkx.Graph, patients_0: List, iterations: int) -> Set:
    total_infected = set(patients_0)
    susceptible = set(graph.nodes) - total_infected

    c_v = {}
    for node in graph.nodes:
        c_v[node] = 0

    for i in range(iterations):
        infected_neighbors_dict = propagate_dict_LTM(graph, total_infected)
        new_infected = set()

        for node in susceptible:
            if node in infected_neighbors_dict:
                sum_of_weights = sum([graph[node][neighbor]['w'] for neighbor in infected_neighbors_dict[node]])
                if CONTAGION * sum_of_weights >= 1 + c_v[node]:
                    new_infected.add(node)
                all_node_neighbors = list(graph.neighbors(node))
                c_v[node] = len(infected_neighbors_dict[node]) / len(all_node_neighbors)

        susceptible = susceptible - new_infected
        total_infected = total_infected.union(new_infected)

    return total_infected


def ICM(graph: networkx.Graph, patients_0: List, iterations: int) -> [Set, Set]:
    degrees = graph.degree()

    NI = patients_0[:]

    # Patients Dying of Lethality round 0
    lethal_deaths = np.random.uniform(0, 1, len(NI)) < LETHALITY
    died = set(np.extract(lethal_deaths, NI))
    NI = list(set(NI) - died)

    total_infected = set(NI)
    total_deceased = set(died)

    targets = propagate_dict(graph, NI, total_infected, total_deceased)
    concern_variables = {target: 0 for target in targets}

    for i in range(iterations):
        # Infect
        NI = infect(graph, targets, concern_variables)

        # Patients Dying of Lethality
        lethal_deaths = np.random.uniform(0, 1, len(NI)) < LETHALITY
        died = set(np.extract(lethal_deaths, NI))

        # Update Sets
        NI = list(set(NI) - died)
        last_round_infected = total_infected.copy()
        last_round_deceased = total_deceased.copy()
        total_deceased = total_deceased.union(died)
        total_infected = total_infected.union(set(NI))

        targets = propagate_dict(graph, NI, total_infected, total_deceased)
        concern_variables = calculate_concern_variables(graph, targets, last_round_infected,
                                                        last_round_deceased, degrees)

    return total_infected, total_deceased


def propagate(graph, NI):
    targets = []
    for node in NI:
        targets += graph.neighbors(node)
    return targets

def propagate_dict(graph, NI, total_infected, total_deceased):
    targets = {}
    for node in NI:
        for neighbor in graph.neighbors(node):
            if neighbor not in total_infected and neighbor not in total_deceased:
                if neighbor not in targets:
                    targets[neighbor] = [node]
                else:
                    targets[neighbor].append(node)
    return targets

def propagate_dict_LTM(graph, NI):
    targets = {}

    for node in NI:
        for neighbor in graph.neighbors(node):

            if neighbor not in targets:
                targets[neighbor] = [node]
            else:
                targets[neighbor].append(node)

    return targets



def calculate_concern_variables(graph, targets, total_infected, total_deceased, degrees):
    concern_variables = {}
    for target in targets:
        neighbors = set(graph.neighbors(target))
        concern_variable = (len(total_infected.intersection(neighbors)) +
                            3 * len(total_deceased.intersection(neighbors))) / len(neighbors)
        concern_variables[target] = min(1, concern_variable)
    return concern_variables


def infect(graph, targets, concern_variables):
    NI = []
    for target in targets:
        concern_variable = concern_variables[target]
        for neighbor in targets[target]:
            infection_probability = min(1, CONTAGION * graph.edges[(target, neighbor)]['w'] *
                                        (1 - concern_variable))
            infected = np.random.uniform() < infection_probability
            if infected:
                NI.append(target)
                break
    return NI


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
    lethailities = (.05, .15, .3, .5, .7)
    mean_deaths = {l: 0 for l in lethailities}
    mean_infected = {l: 0 for l in lethailities}
    for l in lethailities:
        LETHALITY = l
        for iteration in range(30):
            G = copy.deepcopy(graph)
            patients_0 = np.random.choice(list(G.nodes), size=50, replace=False, p=None)
            infected, deceased = ICM(graph, patients_0, t)
            mean_deaths[l] += len(deceased) / 30
            mean_infected[l] += len(infected) / 30
    return mean_deaths, mean_infected


def plot_lethality_effect(mean_deaths: Dict, mean_infected: Dict):
    plt.plot(mean_infected.keys(), mean_infected.values(), label='Mean Infected', color='b')
    plt.plot(mean_deaths.keys(), mean_deaths.values(), label='Mean Deaths', color='r')
    plt.xlabel('Lethality')
    plt.ylabel('Mean')
    plt.title('Lethality Effect')
    plt.legend()
    plt.show()


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


def check_measurement_effectiveness(graph, measurement, reverse=True, iters=5, subgraph_size=5000, num_0=50):
    np.random.seed(0)
    to_remove = len(graph.nodes) - subgraph_size
    subgraph_nodes = np.random.choice(list(graph.nodes), size=to_remove, replace=False, p=None)
    graph.remove_nodes_from(subgraph_nodes)

    measurement_result = measurement(graph)
    top_50 = sorted(measurement_result, reverse=reverse, key=lambda x: measurement_result[x])[:50]
    graph.remove_nodes_from(top_50)



    mean_infected = 0
    mean_deceased = 0
    for _ in range(iters):
        patients_0 = np.random.choice(list(graph.nodes), size=num_0, replace=False, p=None)
        infected, deceased = ICM(graph, patients_0, 6)
        mean_infected += len(infected) / iters
        mean_deceased += len(deceased) / iters

    return mean_infected, mean_deceased


"Global Hyper-parameters"
# CONTAGION = 1
# LETHALITY = .15
patients_0 = [19091, 13254, 5162, 25182, 10872, 6414, 4561, 11881, 1639, 18414, 24468, 9619, 20685, 4033, 14943, 26707, 6675, 16707, 212, 20876, 21798, 17518, 22654, 4914, 21821, 362, 17490, 8472, 23871, 3003, 17531, 20946, 19839, 18587, 17219, 10955, 21184, 24798, 26899, 8370, 17076, 19322, 8734, 1308, 15840, 21292, 1493, 26184, 25897, 6864]

CONTAGION = 0.8
LETHALITY = .2

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
    for measurment in [networkx.degree_centrality,
                       networkx.eigenvector_centrality_numpy,
                       networkx.katz_centrality_numpy,
                       networkx.closeness_centrality,
                       networkx.current_flow_closeness_centrality,
                       networkx.information_centrality,
                       networkx.betweenness_centrality,
                       networkx.current_flow_betweenness_centrality,
                       networkx.load_centrality,
                       networkx.harmonic_centrality,
                       networkx.second_order_centrality,
                       networkx.trophic_levels,
                       networkx.voterank]:
        clone = G3.copy()
        try:
            mean_infected, mean_deceased = check_measurement_effectiveness(clone, measurment, iters=1, subgraph_size=5000,
                                                                           num_0=50)
            print(measurment, mean_infected, mean_deceased)
        except Exception as e:
            print(f"Failed {measurment}, {e}")
        # i = 0
        # sub_graph = G3.subgraph(list(G3.neighbors(i)) + [i])
        # pos = networkx.spring_layout(sub_graph)
        # networkx.draw_networkx(sub_graph, pos)
        # labels = networkx.get_edge_attributes(sub_graph, 'w')
        # labels = {i: round(j, 4) for (i, j) in labels.items()}
        # networkx.draw_networkx_edge_labels(sub_graph, pos, edge_labels=labels)
        # # networkx.draw_networkx_edges(sub_graph, sub_graph.edges)
        # plt.show()

    # ICM(sub_graph, [0], 2)

    #
    # mean_infected = 0
    # mean_deceased = 0
    # iters = 30
    # for i in range(iters):
    #     # infected, deceased = ICM(G3, patients_0[:50], 6)
    #     infected, deceased = ICM(G3, patients_0[:20], 4)
    #     mean_infected += len(infected)
    #     mean_deceased += len(deceased)
    #
    # mean_infected /= iters
    # mean_deceased /= iters
    #
    # print(mean_infected)
    # print(mean_deceased)

    # mean_deaths, mean_infections = compute_lethality_effect(G3, 6)
    # plot_lethality_effect(mean_deaths, mean_infections)
    # CONTAGION = 1
    # print(len(LTM(graph=G3, patients_0=patients_0[:50], iterations=6)))
    # print(len(LTM(graph=G3, patients_0=patients_0[:48], iterations=6)))
    # print(len(LTM(graph=G3, patients_0=patients_0[:30], iterations=6)))
    #
    # CONTAGION = 1.05
    # print(len(LTM(graph=G3, patients_0=patients_0[:30], iterations=6)))
    # print(len(LTM(graph=G3, patients_0=patients_0[:20], iterations=6)))