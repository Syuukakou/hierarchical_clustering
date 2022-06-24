from cmath import inf
import scipy.sparse as sps
import sys
import numpy as np
from itertools import combinations
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
from datetime import datetime
import re
sys.path.append("/home/syuu/Project/nict_clustering")


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
            linkage_matrix[mat_index, :] = [c.index(min_i), c.index(min_j), min_dist, 2]
            cluster_dict[str([min_i, min_j])] = I
        
        if isinstance(min_i, list) and not isinstance(min_j, list):
            i = str(min_i)
            min_i.append(min_j)
            c.append(min_i)
            linkage_matrix[mat_index, :] = [c.index(min_j), cluster_dict[i], min_dist, len(min_i)]
            cluster_dict[str(min_i)] = I

        if not isinstance(min_i, list) and isinstance(min_j, list):
            j = str(min_j)
            min_j.append(min_i)
            c.append(min_j)
            linkage_matrix[mat_index, :] = [c.index(min_i), cluster_dict[j], min_dist, len(min_j)]
            cluster_dict[str(min_j)] = I

        if isinstance(min_i, list) and isinstance(min_j, list):
            i = str(min_i)
            j = str(min_j)
            min_i.extend(min_j)
            c.append(min_i)
            linkage_matrix[mat_index, :] = [cluster_dict[i], cluster_dict[j], min_dist, len(min_i)]
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

def plot_dendrogram():
    """
    func
    """
    label_file_path=r"Files\input_data\avcalss_based_formatted_label_gafgyt_coinminer_sabsik_hajime_tsunami_labels.txt"
    labels = []
    with open(label_file_path, "r", encoding='utf-8') as f:
        for line in f:
            labels.append(line.strip("\n"))

    label_with_index = []
    for l in labels:
        l_index = labels.index(l)
        splitted = re.split("[_ : . /]", l)
        label_with_index.append(
            splitted[0] + "_" + str(l_index) + "_" + splitted[-1].strip("'")
        )
    linkage_matrix = np.load(r"Files\input_data\linkage_matrix.npy")

    fig, ax = plt.subplots(figsize=(80, 60))
    r = dendrogram(
        linkage_matrix,
        labels=label_with_index,
        leaf_font_size=5, ax=ax,
        leaf_rotation=90,
        show_leaf_counts=False)
    
    # https://stackoverflow.com/questions/68122395/how-to-color-a-dendrograms-labels-according-to-defined-groups-in-python
    ## We get the color of leaves from the scipy dendogram docs
    # The key is called "leaves_color_list". We iterate over the list of these colors and set colors for our leaves
    # Please note that this parameter ("leaves_color_list") is different from the "color_list" which is the color of links
    # (as shown in the picture)
    # For the latest names of these parameters, please refer to scipy docs
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.dendrogram.html
    xtick_labels_text = plt.gca().get_xticklabels()
    for leaf, leaf_color in zip(xtick_labels_text, r["leaves_color_list"]):
        leaf.set_color(leaf_color)
    xtick_labels = [i.get_text() for i in xtick_labels_text]
    # plt.axhline(y=0.615, color='r', ls='-')

    colormap = {
        "gafgyt": 'red',
        "coinminer": 'black',
        "oinminer": 'black',
        "sabsik": "olive",
        "hajime": "maroon",
        "tsunami": "blue",
        "mirai": "orange"
    }

    for label in ax.xaxis.get_ticklabels():
        label_text = label.get_text()
        keys = [i for i in list(colormap.keys()) if i in label_text]
        label_index = ax.xaxis.get_ticklabels().index(label)
        if len(keys) > 0:
            ax.xaxis.get_ticklabels()[label_index].set_color(colormap[keys[0]])
        else:
            ax.xaxis.get_ticklabels()[label_index].set_color("pink")


    plt.tight_layout()
    plt.title("No Truncate_mode", fontsize=50)
    ax.tick_params(axis="y", labelsize=20)

    plt.savefig("hcluster/test4.pdf", format='pdf', bbox_inches='tight')


if __name__ == "__main__":
    start = datetime.now()
    # test_clustering()
    plot_dendrogram()
    print(f"END: {datetime.now() - start}")