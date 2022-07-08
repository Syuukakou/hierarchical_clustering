import json
import os
import scipy.sparse as sps
import sys
import numpy as np
from itertools import combinations
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
from datetime import datetime
from bitarray import bitarray
from bitarray.util import count_and
import re
sys.path.append("/home/syuu/Project/nict_clustering")
from numba import njit

def count_abba_i(cluster0, cluster1):
    # bitarray
    c0 = (cluster0).count(1)
    c1 = (cluster1).count(1)
    ab = count_and(cluster0, ~cluster1)
    ba = count_and(~cluster0, cluster1)
    intersection = count_and(cluster0, cluster1)
    return c0, c1, ab, ba, intersection

def calculate_distance(cluster0, cluster1, alpha, beta):
    """
    Assume two sets `A, B`, and `|B| <= |A|`
    `x = |I| / |B|`
    `logistic: f(x) = 1/(1+e^(-\alpha * (x-0.5)))`
    `s_shaped: f(x) = 1 - \franc{1}{1 + (\frac{x}{1-x})^{\alpha}}}`
    ------------------------------------
    `y = 1/2 * (|A-B|/|A| + |B-A|/ |B|)`
    `g(y) = 1 - y^beta`
    ------------------------------------
    `Similarity = f(x) * g(y)`

    Args:
        csr_matrix (_type_): _description_
        v0 (_type_): _description_
        v1 (_type_): _description_
    """
    # start_boolean = datetime.now()
    cluster0_sum, cluster1_sum, ab, ba, intersection = count_abba_i(cluster0, cluster1)
    # print(f"boolean: {datetime.now() - start_boolean}")

    # start_calcu = datetime.now()
    set_distance = calculate_setdistance(cluster0_sum, cluster1_sum, ab, ba, intersection, alpha, beta)
    # print(f"calculate: {datetime.now() - start_calcu}")

    return set_distance

@njit
def calculate_setdistance(cluster0_sum, cluster1_sum, ab, ba, intersection, alpha, beta):
    """
    Use numba to speed up calculation

    Args:
        cluster0_sum (_type_): _description_
        cluster1_sum (_type_): _description_
        ab (_type_): _description_
        ba (_type_): _description_
        intersection (_type_): _description_
        min_b (_type_): _description_
        alpha (_type_): _description_
        beta (_type_): _description_

    Returns:
        _type_: _description_
    """
    i_min_b = intersection / min(cluster0_sum, cluster1_sum)

    # i_factor = calculate_i_min_b(i_min_b, alpha)
    if i_min_b == 1:
        i_factor = 1
    elif i_min_b == 0:
        i_factor = 0
    else:
        # s_shaped
        # i_factor = 1 - 1 / (1 + np.power(i_min_b / (1 - i_min_b), alpha))
        # logistic
        i_factor = 1 / (1 + np.exp(-alpha * (i_min_b - 0.5)))

    c = (ab / cluster0_sum + ba / cluster1_sum) / 2
    c_factor = 1 - np.power(c, beta)
    set_distance = 1 - i_factor * c_factor

    return set_distance

def multiprocess_cal(pair, cluster_dict):
    v0 = pair[0]
    v1 = pair[1]
    # start_calculate = datetime.now()
    pdist = calculate_distance(
        cluster0=cluster_dict[str(v0)][1],
        cluster1=cluster_dict[str(v1)][1],
        # cluster0=np.array(list(cluster_dict[str(v0)][1])).astype(bool),
        # cluster1=np.array(list(cluster_dict[str(v1)][1])).astype(bool),
        # logistic: 12, s_shaped: 5
        alpha=12,
        beta=3
    )

    return [pair, pdist]


def find_min_dist(cluster_dict, c, pair_dist_dict, logistic_alpha=12, s_shaped_alpha=5):
    """_summary_

    Args:
        c (_type_): _description_
    """
    min_dist = np.inf
    min_i = 0
    min_j = 0
    # for pair in tqdm(combinations(c, 2), total=(len(c) * len(c)-1)/2):
    for pair in combinations(c, 2):
        v0 = pair[0]
        v1 = pair[1]
        if (str(v0), str(v1)) not in pair_dist_dict:
            # start_calculate = datetime.now()
            pdist = calculate_distance(
                cluster0=cluster_dict[str(v0)][1],
                cluster1=cluster_dict[str(v1)][1],
                # cluster0=np.array(list(cluster_dict[str(v0)][1])).astype(bool),
                # cluster1=np.array(list(cluster_dict[str(v1)][1])).astype(bool),
                # logistic: 12, s_shaped: 5
                alpha=logistic_alpha,
                beta=3
            )
            pair_dist_dict[(str(v0), str(v1))] = pdist
            # print("---------------------------")
            
        else:
            pdist = pair_dist_dict[(str(v0), str(v1))]
        
        if pdist < min_dist:
            min_dist = pdist
            min_i = v0
            min_j = v1
        # print(f"Calculation time: {datetime.now() - start_calculate}")
        
    return min_i, min_j, min_dist


def bitarray_union(clusters_dict: dict, key_0: str, key_1: str):
    """_summary_

    Args:
        clusters_dict (dict): _description_
        key_0 (str): _description_
        key_1 (str): _description_

    Returns:
        _type_: _description_
    """
    return (clusters_dict[key_0][1] | clusters_dict[key_1][1])


def set_dist_clustering(npz_filepath: str,
                        label_file_path: str,
                        linkage_matrix_save_path: str,
                        save_name: str):
    """
    Reference from
    https://www.saedsayad.com/clustering_hierarchical.htm
    """
    # label_file_path="/home/syuu/Project/nict_clustering/latest_analysis_date/clustering/gafgyt_coinminer_sabsik_hajime_tsunami/input_data/avcalss_based_formatted_label_gafgyt_coinminer_sabsik_hajime_tsunami_labels.txt"
    labels = []
    with open(label_file_path, "r", encoding='utf-8') as f:
        for line in f:
            labels.append(line.strip("\n"))

    # npz_filepath="/home/syuu/Project/nict_clustering/latest_analysis_date/clustering/gafgyt_coinminer_sabsik_hajime_tsunami/input_data/avcalss_based_formatted_label_gafgyt_coinminer_sabsik_hajime_tsunami_CSR.npz"
    csr_matrix = sps.load_npz(npz_filepath)

    print(f"{datetime.now()}: Convert Sparse matrix to dense...")
    print(f"{datetime.now()}: Convert dense matrix to bitarray")
    # bit_arr = ()
    # for i in range(0, csr_matrix.shape[0]):
    #     row_bit = bitarray(csr_matrix.getrow(i).toarray().tolist())
    #     bit_arr = bit_arr + (row_bit, )
    bit_arr = (bitarray(csr_matrix[i].toarray().tolist()[0]) for i in csr_matrix.shape[0])
    # mat_dense = csr_matrix.astype(np.int8).A
    # bit_arr = (bitarray((i).tolist()) for i in mat_dense)
    # with open("latest_analysis_date/clustering/20220620/input_data/20220620_bitarray.txt", "a+", encoding='utf-8') as f:
    #     for i in bit_arr:
    #         i.tofile(f=f)
   
    # c = []
    # c = numba.typed.List(list(range(csr_matrix.shape[0])))
    print(f"{datetime.now()}: Clustering")
    c = list(range(csr_matrix.shape[0]))
    I = len(c)

    linkage_matrix = np.zeros((len(c)-1, 4))
    mat_index = 0
    print("Init Dict:...")
    pair_dist_dict = {}
    cluster_dict = {str(i): (i, j) for (i, j) in zip(c, bit_arr)}
    print("---------------------------------------------------------")
    # logistic_alpha, s_shaped_alpha, beta = 12, 5, 3

    while len(c) > 1:
        circle_start = datetime.now()
        print(f"Clusters: {len(c)}")
        print("Finding min pdist...")
        min_i, min_j, min_dist = find_min_dist(cluster_dict, c, pair_dist_dict)

        print(f"min_i: {min_i}, min_j: {min_j}, dist: {min_dist}")
        # print(f"{labels[min_i]}, {labels[min_j]}, {min_dist}")

        if not isinstance(min_i, list) and not isinstance(min_j, list):
            # i_index = c.index(min_i)
            # j_index = c.index(min_j)
            # remove min_i and min_j from c
            print(f"Remove {min_i}, {min_j}, {min_dist}, {I}")
            c.remove(min_i)
            c.remove(min_j)
            # c = remove_list_element(c, min_i, min_j)
            # add [min_i, min_j] to c
            c.append([min_i, min_j])
            # c = append_element(c, [min_i, min_j])

            linkage_matrix[mat_index, :] = [min_i, min_j, min_dist, 2]
            cluster_dict[str([min_i, min_j])] = (I, bitarray_union(cluster_dict, str(min_i), str(min_j)))
        
        elif isinstance(min_i, list) and not isinstance(min_j, list):
            i = str(min_i)
            # j_index = c.index(min_j)
            print(f"Remove {min_i}, {min_j}, {min_dist}, {I}")
            c.remove(min_j)
            c.remove(min_i)

            min_i.append(min_j)
            c.append(min_i)
            linkage_matrix[mat_index, :] = [min_j, cluster_dict[i][0], min_dist, len(min_i)]
            cluster_dict[str(min_i)] = (I, bitarray_union(cluster_dict, i, str(min_j)))

        elif not isinstance(min_i, list) and isinstance(min_j, list):
            j = str(min_j)
            # i_index = c.index(min_i)
            print(f"Remove {min_i}, {min_j}, {min_dist}, {I}")
            c.remove(min_i)
            c.remove(min_j)

            min_j.append(min_i)
            c.append(min_j)
            linkage_matrix[mat_index, :] = [min_i, cluster_dict[j][0], min_dist, len(min_j)]
            cluster_dict[str(min_j)] = (I, bitarray_union(cluster_dict, j, str(min_i)))

        elif isinstance(min_i, list) and isinstance(min_j, list):
            i = str(min_i)
            j = str(min_j)
            print(f"Remove {min_i}, {min_j}, {min_dist}, {I}")
            c.remove(min_i)
            c.remove(min_j)

            min_i.extend(min_j)
            c.append(min_i)
            print(cluster_dict[j][0])
            linkage_matrix[mat_index, :] = [cluster_dict[i][0], cluster_dict[j][0], min_dist, len(min_i)]
            cluster_dict[str(min_i)] = (I, bitarray_union(cluster_dict, i, j))
        
        print(linkage_matrix[mat_index, :])
        print(len([i for i in c if isinstance(i, list)]))

        mat_index += 1
        I += 1
        print(f"Clusters: {len(c)}")
        print(f"Time consuming: {datetime.now() - circle_start}")
        print("---------------------------------------------------------")
    
    print(linkage_matrix)
    # 
    # /home/syuu/Project/nict_clustering/hcluster/files/s_shaped
    np.save(f"{linkage_matrix_save_path}/{save_name}.npy", linkage_matrix)


