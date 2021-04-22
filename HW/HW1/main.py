import time
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
    plt.xlabel('Degree')
    plt.ylabel('Number of Nodes')
    plt.title("Degree Histogram")
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
    plt.plot(list(mean_infected.keys()), list(mean_infected.values()), label='Mean Infected', color='b')
    plt.plot(list(mean_deaths.keys()), list(mean_deaths.values()), label='Mean Deaths', color='r')
    plt.xlabel('Lethality')
    plt.ylabel('Mean')
    plt.title('Lethality Effect')
    plt.legend()
    plt.show()


def choose_who_to_vaccinate(graph: networkx.Graph) -> List:
    time_threshold = 55
    iters = 5
    sampling_iters = 1
    num_0 = 50
    lower_bound = 0

    start = time.time()

    best_50 = np.random.choice(list(graph.nodes), size=50, replace=False, p=None)
    best_mean_infected = float('inf')
    best_upper_bound = float('inf')
    clustering_coefficients = networkx.clustering(graph)
    degrees = networkx.degree(graph)

    for upper_bound in [0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055]:
        if time.time() - start > time_threshold:
            return best_50
        filtered_coefficients = {c: clustering_coefficients[c] for c in clustering_coefficients if
                                 clustering_coefficients[c] <= upper_bound and clustering_coefficients[c] > lower_bound}
        filtered_degrees = {c: degrees[c] for c in filtered_coefficients}
        top_50 = sorted(filtered_degrees.keys(), key=lambda x: filtered_degrees[x], reverse=True)[:50]
        clone = G3.copy()
        clone.remove_nodes_from(top_50)

        np.random.seed(0)

        mean_infected = 0
        for _ in range(iters):
            patients_0 = np.random.choice(list(clone.nodes), size=num_0, replace=False, p=None)
            infected, deceased = ICM(clone, patients_0, 6)
            mean_infected += len(infected) / iters

        if mean_infected < best_mean_infected:
            print(mean_infected)
            print(upper_bound)
            best_mean_infected = mean_infected
            best_upper_bound = upper_bound
            best_50 = top_50

    filtered_coefficients = {c: clustering_coefficients[c] for c in clustering_coefficients if
                             clustering_coefficients[c] <= best_upper_bound and clustering_coefficients[c] > lower_bound}
    filtered_degrees = {c: degrees[c] for c in filtered_coefficients}
    top_150 = sorted(filtered_degrees.keys(), key=lambda x: filtered_degrees[x], reverse=True)[:80]
    top_150_degrees = [filtered_degrees[node] for node in top_150]
    weights = np.exp(top_150_degrees) / np.sum(np.exp(top_150_degrees), axis=0)

    print('started sampling')
    while time.time() - start < time_threshold:
        to_remove = np.random.choice(top_150, 50, p=weights, replace=False)
        clone = G3.copy()
        clone.remove_nodes_from(to_remove)

        np.random.seed(0)

        mean_infected = 0
        for _ in range(sampling_iters):
            patients_0 = np.random.choice(list(clone.nodes), size=num_0, replace=False, p=None)
            infected, deceased = ICM(clone, patients_0, 6)
            mean_infected += len(infected) / sampling_iters

        if mean_infected < best_mean_infected:
            print(mean_infected)
            best_mean_infected = mean_infected
            best_50 = to_remove

    return best_50


def check_measurement_effectiveness(graph, measurement, reverse=True, iters=5, subgraph_size=5000, num_0=50):
    np.random.seed(0)

    mean_infected = 0
    mean_deceased = 0

    for _ in range(iters):
        clone = graph.copy()
        # to_remove = len(clone.nodes) - subgraph_size
        # subgraph_nodes = np.random.choice(list(clone.nodes), size=to_remove, replace=False, p=None)
        # clone.remove_nodes_from(subgraph_nodes)

        measurement_result = measurement(clone)
        measurement_result = {i: j for (i, j) in measurement_result}
        top_50 = sorted(measurement_result, reverse=reverse, key=lambda x: measurement_result[x])[:50]
        clone.remove_nodes_from(top_50)


        patients_0 = np.random.choice(list(clone.nodes), size=num_0, replace=False, p=None)
        infected, deceased = ICM(clone, patients_0, 6)
        mean_infected += len(infected) / iters
        mean_deceased += len(deceased) / iters

    return mean_infected, mean_deceased


"Global Hyper-parameters"
# CONTAGION = 1
# LETHALITY = .15
patients0 = [19091, 13254, 5162, 25182, 10872, 6414, 4561, 11881, 1639, 18414, 24468, 9619, 20685, 4033, 14943, 26707, 6675, 16707, 212, 20876, 21798, 17518, 22654, 4914, 21821, 362, 17490, 8472, 23871, 3003, 17531, 20946, 19839, 18587, 17219, 10955, 21184, 24798, 26899, 8370, 17076, 19322, 8734, 1308, 15840, 21292, 1493, 26184, 25897, 6864]

CONTAGION = 0.8
LETHALITY = .15

if __name__ == "__main__":
    filename_1 = "PartA1.csv"
    filename_2 = "PartA2.csv"
    filename_3 = "PartB-C.csv"

    #########################
    ######## PART A #########
    #########################

    # Building Graphs
    G1 = build_graph(filename=filename_1)
    G2 = build_graph(filename=filename_2)
    G3 = build_graph(filename=filename_3)

    # Q2 Calculating degree histograms
    histogram_1 = calc_degree_histogram(G1)
    histogram_2 = calc_degree_histogram(G2)
    histogram_3 = calc_degree_histogram(G3)

    # Q3 Plotting degree histograms
    plot_degree_histogram(histogram_1)
    plot_degree_histogram(histogram_2)
    plot_degree_histogram(histogram_3)

    # Q5 Calculating clustering coefficient
    print(f"Clustering coefficient of G1: {clustering_coefficient(G1)}")
    print(f"Clustering coefficient of G2: {clustering_coefficient(G2)}")

    #########################
    ######## PART B #########
    #########################

    # Q5 Calculating and plotting lethality effect
    mean_deaths, mean_infections = compute_lethality_effect(G3, 6)
    plot_lethality_effect(mean_deaths, mean_infections)

    # Part B tests

    # LTM
    # CONTAGION = 1
    print(f"LTM check:")
    print(len(LTM(G3, patients0[:50], 6)))
    print(len(LTM(G3, patients0[:48], 6)))
    print(len(LTM(G3, patients0[:30], 6)))
    #
    # CONTAGION = 1.05
    print(len(LTM(G3, patients0[:30], 6)))
    print(len(LTM(G3, patients0[:20], 6)))

    # ICM
    CONTAGION = 0.8
    LETHALITY = 0.2
    print(f"ICM check:")
    infected, deceased = ICM(G3, patients0[:50], 6)
    print(len(infected), len(deceased))
    infected, deceased = ICM(G3, patients0[:20], 4)
    print(len(infected), len(deceased))

    #########################
    ######## PART C #########
    #########################
    print(f"Part C:")
    best_50 = choose_who_to_vaccinate(G3)
    print(f"Best 50 to vaccinate in G3: {best_50}")