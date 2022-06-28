from cmath import inf
import scipy.sparse as sps
import sys
import numpy as np
from itertools import combinations
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
from datetime import datetime
import re
import json
sys.path.append("/home/syuu/Project/nict_clustering")
# from Src.functions import plot_pairwise_intersection_based_matrix
from Clustering_Plot_methods.functions import plot_pairwise_intersection_based_matrix, i_min_ab_a, one_hot_encoding

def calculate_distance(csr_matrix, v0, v1, alpha, beta):
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
    start_time = datetime.now()
    # print(f"v0: {v0}, v1: {v1}")
    cluster0 = csr_matrix[v0, :].sum(axis=0).astype(bool)
    cluster1 = csr_matrix[v1, :].sum(axis=0).astype(bool)
    
    cluster0_sum = cluster0.sum()
    cluster1_sum = cluster1.sum()

    min_b = min(cluster0_sum, cluster1_sum)
    # intersection = (cluster0 & cluster1).sum()
    intersection = np.multiply(cluster0, cluster1).sum()
    i_min_b = intersection / min_b
    if i_min_b == 1:
        i_factor = 1
    elif i_min_b == 0:
        i_factor = 0
    else:
        # s_shaped
        # i_factor = 1 - 1 / (1 + np.power(i_min_b / (1 - i_min_b), alpha))
        # logistic
        i_factor = 1 / (1 + np.exp(-alpha * (i_min_b - 0.5)))
    
    # ab = np.greater(cluster0 > cluster1).sum()
    # ab = (cluster0 > cluster1).sum()
    # ba = np.less(cluster0 < cluster1).sum()
    # ba = (cluster0 < cluster1).sum()
    c = (np.greater(cluster0, cluster1).sum() / cluster0_sum + np.less(cluster0, cluster1).sum() / cluster1_sum) / 2
    # c = (ab / cluster0_sum + ba / cluster1_sum) / 2
    c_factor = 1 - np.power(c, beta)
    print(f"Calculation time: {datetime.now() - start_time}")

    return 1 - i_factor * c_factor

def pdist_mat(csr_matrix, c, logistic_alpha, s_shaped_alpha, beta):
    """_summary_

    Args:
        csr_matrix (_type_): _description_
        c (_type_): _description_
    """
    mat_list = [sps.csr_matrix(csr_matrix[i, :].sum(axis=0)) for i in c]
    # print(mat_list[0])
    mat = sps.vstack(mat_list)
    intersection, logisti_similarity = i_min_ab_a(
        csr_matrix=mat,
        logistic_alpha=logistic_alpha,
        s_shaped_alpha=s_shaped_alpha,
        beta=beta,
        s_shaped_func=False
    )
    logistic_dist = 1 - logisti_similarity
    # s_shaped_dist = 1 - s_shaped_similarity

    logistic_dist[np.tril_indices(logistic_dist.shape[0], -1)] = np.inf
    np.fill_diagonal(logistic_dist, np.inf)

    # s_shaped_dist[np.tril_indices(s_shaped_dist.shape[0], -1)] = np.inf
    # np.fill_diagonal(s_shaped_dist, 0)

    logistic_min_ij = np.unravel_index(logistic_dist.argmin(), logistic_dist.shape)
    # s_shaped_min_ij = np.unravel_index(s_shaped_dist.argmin(), logistic_dist.shape)
    # , (c[s_shaped_min_ij[0]], c[s_shaped_min_ij[1]], s_shaped_dist.argmin())

    return (c[logistic_min_ij[0]], c[logistic_min_ij[1]], logistic_dist.argmin())


def find_min_dist(csr_matrix, c, logistic_alpha=12, s_shaped_alpha=5):
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
            # logistic: 12, s_shaped: 5
            alpha=logistic_alpha,
            beta=3
        )
        if pdist < min_dist:
            min_dist = pdist
            min_i = v0
            min_j = v1
    return min_i, min_j, min_dist

def set_dist_clustering(npz_filepath: str,
                        label_file_path: str,
                        linkage_matrix_save_path: str):
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
   
    # c = []
    c = list(range(csr_matrix.shape[0]))

    I = len(c)

    linkage_matrix = np.zeros((len(c)-1, 4))
    mat_index = 0
    cluster_dict = {}
    # logistic_alpha, s_shaped_alpha, beta = 12, 5, 3

    while len(c) > 1:
        circle_start = datetime.now()
        print(f"Clusters: {len(c)}")
        print("Finding min pdist...")
        min_i, min_j, min_dist = find_min_dist(csr_matrix, c)

        # logistic_min_dist = pdist_mat(
        #     csr_matrix=csr_matrix,
        #     c=c,
        #     logistic_alpha=logistic_alpha,
        #     s_shaped_alpha=s_shaped_alpha,
        #     beta = beta
        # )
        # min_i, min_j, min_dist = logistic_min_dist[0], logistic_min_dist[1], logistic_min_dist[2]
        print(f"min_i: {min_i}, min_j: {min_j}, dist: {min_dist}")
        # print(f"{labels[min_i]}, {labels[min_j]}, {min_dist}")

        if not isinstance(min_i, list) and not isinstance(min_j, list):
            # i_index = c.index(min_i)
            # j_index = c.index(min_j)
            # remove min_i and min_j from c
            print(f"Remove {min_i}, {min_j}, {min_dist}, {I}")
            c.remove(min_i)
            c.remove(min_j)
            # add [min_i, min_j] to c
            c.append([min_i, min_j])

            linkage_matrix[mat_index, :] = [min_i, min_j, min_dist, 2]
            cluster_dict[str([min_i, min_j])] = I
        
        elif isinstance(min_i, list) and not isinstance(min_j, list):
            i = str(min_i)
            # j_index = c.index(min_j)
            print(f"Remove {min_i}, {min_j}, {min_dist}, {I}")
            c.remove(min_j)
            c.remove(min_i)

            min_i.append(min_j)
            c.append(min_i)
            linkage_matrix[mat_index, :] = [min_j, cluster_dict[i], min_dist, len(min_i)]
            cluster_dict[str(min_i)] = I

        elif not isinstance(min_i, list) and isinstance(min_j, list):
            j = str(min_j)
            # i_index = c.index(min_i)
            print(f"Remove {min_i}, {min_j}, {min_dist}, {I}")
            c.remove(min_i)
            c.remove(min_j)

            min_j.append(min_i)
            c.append(min_j)
            linkage_matrix[mat_index, :] = [min_i, cluster_dict[j], min_dist, len(min_j)]
            cluster_dict[str(min_j)] = I

        elif isinstance(min_i, list) and isinstance(min_j, list):
            i = str(min_i)
            j = str(min_j)
            print(f"Remove {min_i}, {min_j}, {min_dist}, {I}")
            c.remove(min_i)
            c.remove(min_j)

            min_i.extend(min_j)
            c.append(min_i)
            linkage_matrix[mat_index, :] = [cluster_dict[i], cluster_dict[j], min_dist, len(min_i)]
            cluster_dict[str(min_i)] = I
        
        print(linkage_matrix[mat_index, :])
        # c.remove(min_i)
        # c.remove(min_j)
        print([i for i in c if isinstance(i, list)])

        mat_index += 1
        I += 1
        print(f"Clusters: {len(c)}")
        print(f"Time consuming: {datetime.now() - circle_start}")
        print("---------------------------------------------------------")
    
    print(linkage_matrix)
    # 
    # /home/syuu/Project/nict_clustering/hcluster/files/s_shaped
    np.save(f"{linkage_matrix_save_path}/linkage_matrix_logistic.npy", linkage_matrix)


def plot_dendrogram(save_path: str,
                    label_file_path: str,
                    npz_filepath: str,
                    linkage_matrix_filepath: str,
                    family_name_count_json_filepath: str):
    """
    func
    """
    # save_path = "/home/syuu/Project/nict_clustering/hcluster/files/s_shaped"
    # label_file_path="/home/syuu/Project/nict_clustering/latest_analysis_date/clustering/gafgyt_coinminer_sabsik_hajime_tsunami/input_data/avcalss_based_formatted_label_gafgyt_coinminer_sabsik_hajime_tsunami_labels.txt"
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
    linkage_matrix = np.load(linkage_matrix_filepath)
    # linkage_matrix = np.load("/home/syuu/Project/nict_clustering/hcluster/files/s_shaped/linkage_matrix_s_shaped.npy")

    fig, ax = plt.subplots(figsize=(80, 60))
    r = dendrogram(
        linkage_matrix,
        labels=label_with_index,
        distance_sort=False,
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

    # * Create colormap from families
    with open(family_name_count_json_filepath, "r", encoding='utf-8') as f:
        family_names_count = json.load(f)
    family_names = list(family_names_count.keys())
    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0, 1, len(len(family_names_count))))
    colormap = {}
    for c, fam in zip(colors, family_names):
        colormap[fam] = c
    # colormap = {
    #     "gafgyt": 'red',
    #     "coinminer": 'black',
    #     "oinminer": 'black',
    #     "sabsik": "olive",
    #     "hajime": "maroon",
    #     "tsunami": "blue",
    #     "mirai": "orange"
    # }

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

    plt.savefig(f"{save_path}/s_shaped.pdf", format='pdf', bbox_inches='tight')

    # npz_filepath="/home/syuu/Project/nict_clustering/latest_analysis_date/clustering/gafgyt_coinminer_sabsik_hajime_tsunami/input_data/avcalss_based_formatted_label_gafgyt_coinminer_sabsik_hajime_tsunami_CSR.npz"
    csr_matrix = sps.load_npz(npz_filepath)
    intersection = csr_matrix.dot(csr_matrix.T)
    xtick_labels = [i.get_text() for i in xtick_labels_text]

    plot_pairwise_intersection_based_matrix(
        csr_matrix=csr_matrix,
        intersection_csr_matrix=intersection,
        labels=label_with_index,
        xtick_labels=xtick_labels,
        colormap=colormap,
        save_path = save_path,
        keywords_str="all",
        alpha=12,
        method="Set_Dist"
    )

def all_data_preprocess():
    one_hot_encoding(
        json_file_path="Data_preprocess/data/20220620_formatted_label_hash.json",
        save_path="latest_analysis_date/clustering/20220620",
        save_name='20220620'
    )


def test_bytearray():
    """
    test func
    """
    csr_matrix = sps.load_npz("latest_analysis_date/clustering/gafgyt_coinminer_sabsik_hajime_tsunami/input_data/avcalss_based_formatted_label_gafgyt_coinminer_sabsik_hajime_tsunami_CSR.npz")
    mat_dense = csr_matrix.astype(np.int8).A
    bytearray_list = [bytearray(i) for i in mat_dense]
    
if __name__ == "__main__":
    start = datetime.now()
    test_npz_file = "latest_analysis_date/clustering/gafgyt_coinminer_sabsik_hajime_tsunami/input_data/avcalss_based_formatted_label_gafgyt_coinminer_sabsik_hajime_tsunami_CSR.npz"
    test_label_text_file = "latest_analysis_date/clustering/gafgyt_coinminer_sabsik_hajime_tsunami/input_data/avcalss_based_formatted_label_gafgyt_coinminer_sabsik_hajime_tsunami_labels.txt"
    set_dist_clustering(
        npz_filepath=test_npz_file,
        label_file_path=test_label_text_file,
        linkage_matrix_save_path="hcluster/files/linkage_matrix",
        # save_name="linkage_matrix_bytearray"
    )
    # plot_dendrogram()
    # all_data_preprocess()
    print(f"END: {datetime.now() - start}")