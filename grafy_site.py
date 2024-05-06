import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from graspologic.plot import adjplot, matrixplot
import pandas as pd
import plotly.graph_objects as go

def no_coupling(n):
    K = []
    for row in range(n):
        line = []
        for col in range(n):
            line.append(0)
        K.append(line)
    return np.asarray(K)


def small_world(n, k, p, seed):
    # n..pocet neuronu, k..s kolika neurony je propojen, p..pst prepojeni
    G = nx.watts_strogatz_graph(n, k, p, seed)
    return nx.adjacency_matrix(G).todense()


def small_world_outer(n, k, p, seed):
    np.random.seed(15)
    # pravidelna struktura
    K = np.zeros((n, n))
    for i in range(k):
        K = K + np.diag([1] * (n - i), i) + np.diag([1] * i, i - n)
    # prepojeni
    K_random = np.random.random((n, n))
    K_index = np.random.randint(0, n * n, size=(n, n))
    for row in range(n):
        for col in range(n):
            if K[row][col] == 1:
                if K_random[row][col] < p:
                    K[row][col] = 0
                    index = K_index[row][col]
                    K[index // n][index % n] = 1
    return np.asarray(K)

def all_to_all(n):
    K = []
    for row in range(n):
        line = []
        for col in range(n):
            if row == col:
                line.append(0)
            else:
                line.append(1)
        K.append(line)
    return np.asarray(K)


def all_to_all_outer(n):
    return np.ones((n, n))


def random_network(n, p, seed):
    G = nx.erdos_renyi_graph(n, p, seed, directed=False)  # random network
    return nx.adjacency_matrix(G).todense()


def random_network_outer(n, p, seed):
    np.random.seed(seed)
    K = np.random.random((n, n))
    for row in range(n):
        for col in range(n):
            if K[row][col] < p:
                K[row][col] = 1
            else:
                K[row][col] = 0
    return np.asarray(K)


def ring(n):
    if n == 1:
        return np.asarray([[0]])
    K = []
    for row in range(n):
        line = []
        for col in range(n):
            if col == (row-1) % n or col == (row+1) % n:
                line.append(1)
            else:
                line.append(0)
        K.append(line)
    return np.asarray(K)


def ring_outer(n):
    K = np.zeros((n, n))
    K[n-1][0] = 1
    return np.asarray(K)

########################################################################

def generate_K(coupling_type, n, k, p, seed):
    # 1: no coupling, 2: all to all, 3: smallworld, 4: random network, 5: ring
    if coupling_type == 2:
        K = all_to_all(n)
    elif coupling_type == 3:
        K = small_world(n, k, p, seed)
    elif coupling_type == 4:
        K = random_network(n, p, seed)
    elif coupling_type == 5:
        K = ring(n)
    else:
        K = no_coupling(n)
    return K


def generate_outer_K(coupling_type, n, k, p, seed, loc):
    if coupling_type == 2:
        K = all_to_all_outer(n)
    elif coupling_type == 3:
        K = small_world_outer(n, k, p, seed)
    elif coupling_type == 4:
        K = random_network_outer(n, p, seed)
    elif coupling_type == 5:
        K = ring_outer(n)
    else:
        K = no_coupling(n)

    if loc == "U":
        return K
    else:
        return np.transpose(K)

########################################################################

if __name__ == '__main__':
    # POCTY NEURONU
    n_neurons = 10
    n_clusters = 2

    # NASTAVENI - JEDEN CLUSTER
    # epsilon = [[1]]
    # # 1: no coupling, 2: all to all, 3: smallworld, 4: random network, 5: ring
    # coupling = [[2]]
    # k = [[2]]
    # p = [[0.5]]
    # seed = [[10]]

    # NASTAVENI - VICE CLUSTERU
    epsilon = [[1, 1], [1, 1]]
    # 1: no coupling, 2: all to all, 3: smallworld, 4: random network, 5: ring
    coupling = [[2, 3], [3, 2]]
    k = [[2, 3], [3, 2]]
    p = [[0.5, 0.3], [0.3, 0.5]]
    seed = [[10, 10], [10, 10]]


    # VYPOCET MATICE K
    K_list = []
    for row in range(n_clusters):
        line = []
        for col in range(n_clusters):
            if row == col:
                K = epsilon[row][col] * \
                    generate_K(coupling[row][col], n_neurons, k[row][col], p[row][col], int(seed[row][col]))
            elif row < col:
                K = epsilon[row][col] * \
                    generate_outer_K(coupling[row][col], n_neurons, k[row][col], p[row][col], int(seed[row][col]), "U")
            else:
                K = epsilon[row][col] * \
                    generate_outer_K(coupling[row][col], n_neurons, k[row][col], p[row][col], int(seed[row][col]), "L")
            line.append(K)
        K_list.append(line)
    K = np.concatenate([np.concatenate(row, axis=1) for row in K_list], axis=0)


    # GRAF SITE
    plt.rcParams.update({'font.size': 18})
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    G = nx.DiGraph()
    for row in range(len(K)):
        G.add_node(row)
        for col in range(len(K)):
            if K[row][col] == 1:
                G.add_edge(row, col)
    nx.draw(G, ax=ax[1], with_labels=True)

    for row in range(n_clusters*n_neurons):
        for col in range(n_clusters*n_neurons):
            if row >= n_neurons > col:
                K[row][col] = K[row][col]*0.5
            if col >= n_neurons > row:
                K[row][col] = K[row][col]*0.5

    # ADJPLOT

    meta = pd.DataFrame(
        data={'cell_size': np.arange(len(K))},
    )

    adjplot(
        data=K,
        ax=ax[0],
        meta=meta,
        plot_type="heatmap",
        sizes=(5, 5),
    )



    #plt.suptitle('Síť malého světa s $n=20$ neurony s $k=3, p=0.3$')
    plt.suptitle('Síť se dvěma shluky s $n=10$ neurony spřaženými každý s každým\n Shluky vzájemně spřaženy sítí malého světa s $k=3, p=0,3$')
    plt.tight_layout()
    plt.show()
