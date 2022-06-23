from cmath import inf
import scipy.sparse as sps
import sys
import numpy as np
from itertools import combinations
sys.path.append(r"C:\Users\Syuukakou\PycharmProjects\hierarchical_clustering")
from Src.functions import (extract_and_plot,
                replace_with_dict2, check_symmetric,
                plot_pairwise_intersection_based_matrix,
                i_min_ab_a)


def calculate_distance(csr_matrix, v0, v1, alpha, beta):
    """
    Assume two sets `A, B`, and `|B| <= |A|`
    `x = |I| / |B|`
    `logistic: f(x) = 1/(1+e^(-\alpha * (x-0.5)))`
    `s_shaped: f(x) = 1 / (1 + e^(-\alpha * (i_u - 0.5)))`
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
    # print(f"v0: {v0}, v1: {v1}")
    cluster0 = csr_matrix[v0, :].sum(axis=0).astype(bool)
    cluster1 = csr_matrix[v1, :].sum(axis=0).astype(bool)

    cluster0_sum = cluster0.sum()
    cluster1_sum = cluster1.sum()

    min_b = min(cluster0_sum, cluster1_sum)
    intersection = np.multiply(cluster0, cluster1).sum()
    i_min_b = intersection / min_b
    if i_min_b == 1:
        i_factor = 1
    elif i_min_b == 0:
        i_factor = 0
    else:
        i_factor = 1 / (1 + np.exp(-alpha * (i_min_b - 0.5)))

    ab = (cluster0 > cluster1).sum()
    ba = (cluster0 < cluster1).sum()
    c = (ab / cluster0_sum + ba / cluster1_sum) / 2
    c_factor = 1 - np.power(c, beta)

    return 1 - i_factor * c_factor

def find_min_dist(csr_matrix, c):
    """_summary_

    Args:
        c (_type_): _description_
    """
    min_dist = inf
    min_i = 0
    min_j = 0
    for pair in combinations(c, 2):
        v0 = pair[0]
        v1 = pair[1]
        pdist = calculate_distance(
            csr_matrix=csr_matrix,
            v0=v0,
            v1=v1,
            alpha=12,
            beta=3
        )
        if pdist < min_dist:
            min_dist = pdist
            min_i = v0
            min_j = v1
        
    return min_i, min_j, min_dist

def test_clustering():
    """
    https://www.saedsayad.com/clustering_hierarchical.htm
    """
    label_file_path="latest_analysis_date/clustering/gafgyt_coinminer_sabsik_hajime_tsunami/input_data/avcalss_based_formatted_label_gafgyt_coinminer_sabsik_hajime_tsunami_labels.txt"
    labels = []
    with open(label_file_path, "r", encoding='utf-8') as f:
        for line in f:
            labels.append(line.strip("\n"))

    npz_filepath="latest_analysis_date/clustering/gafgyt_coinminer_sabsik_hajime_tsunami/input_data/avcalss_based_formatted_label_gafgyt_coinminer_sabsik_hajime_tsunami_CSR.npz"
    csr_matrix = sps.load_npz(npz_filepath)
   
    c = []
    c = list(range(csr_matrix.shape[0]))

    I = len(c) + 1

    linkage_matrix = np.zeros((len(c)-1, 4))
    mat_index = 0
    cluster_dict = {}

    while len(c) > 1:
        print(f"Clusters: {len(c)}")
        print("Finding min pdist...")
        min_i, min_j, min_dist = find_min_dist(csr_matrix, c)
        print(f"min_i: {min_i}, min_j: {min_j}, dist: {min_dist}")
        # print(f"{labels[min_i]}, {labels[min_j]}, {min_dist}")

        if not isinstance(min_i, list) and not isinstance(min_j, list):
            c.append([min_i, min_j])
            linkage_matrix[mat_index, :] = [c.index(min_i), I, min_dist, 2]
            cluster_dict[str([min_i, min_j])] = I
        
        if isinstance(min_i, list) and not isinstance(min_j, list):
            min_i.append(min_j)
            c.append(min_i)
            linkage_matrix[mat_index, :] = [c.index(min_j), I, min_dist, len(min_i)]
            cluster_dict[str(min_i)] = I

        if not isinstance(min_i, list) and isinstance(min_j, list):
            min_j.append(min_i)
            c.append(min_j)
            linkage_matrix[mat_index, :] = [c.index(min_i), I, min_dist, len(min_j)]
            cluster_dict[str(min_j)] = I

        if isinstance(min_i, list) and isinstance(min_j, list):
            min_i.extend(min_j)
            c.append(min_i)
            linkage_matrix[mat_index, :] = [cluster_dict[min_i], cluster_dict[min_j], min_dist, len(min_i)]
            cluster_dict[str(min_i)] = I
        
        print(f"Remove {min_i}, {min_j}, {min_dist}, {I}")
        print(linkage_matrix[mat_index, :])
        c.remove(min_i)
        c.remove(min_j)
        print([i for i in c if isinstance(i, list)])

        mat_index += 1
        I += 1
        print("---------------------------------------------------------")
    
    print(linkage_matrix)
    np.save("hcluster/linkage_matrix.npy", linkage_matrix)

if __name__ == "__main__":
    test_clustering()