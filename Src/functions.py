"""
1. Sparse matrix: how to get nonzero indices for each row
https://stackoverflow.com/questions/44536107/sparse-matrix-how-to-get-nonzero-indices-for-each-row

2. How can I remove a column from a sparse matrix efficiently?
https://stackoverflow.com/questions/2368544/how-can-i-remove-a-column-from-a-sparse-matrix-efficiently
https://stackoverflow.com/questions/23966923/delete-columns-of-matrix-of-csr-format-in-python

3. finding y index of a sparse 2D matrix by its value in python
https://stackoverflow.com/questions/36366585/finding-y-index-of-a-sparse-2d-matrix-by-its-value-in-python
"""

from cmath import inf
import numpy as np
import re
import scipy.sparse as sps
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from itertools import combinations
import scipy
import pandas as pd
from itertools import combinations
import json
from tabulate import tabulate
import seaborn as sns
from datetime import datetime
import sys
import os
sys.path.append("/home/syuu/Project/nict_clustering")
#pylint: disable=wrong-import-position

def calculate_sparse_matrix_minus_exp(matrix_data, union_length, alpha):
    """_summary_

    Args:
        matrix_data (_type_): _description_
        alpha (_type_): _description_

    Returns:
        _type_: _description_
    """
    print(type(matrix_data))
    matrix_data.data = matrix_data.data / union_length
    # print(matrix_data.data)
    # idata = [- alpha * i for i in matrix_data.data]
    idata = [- alpha * (i - 0.5) for i in matrix_data.data]
    
    matrix_data.data = np.exp(idata)
    # print(matrix_data.data)
    exp_i = matrix_data.A
    # print((exp_i))
    exp_i[exp_i == 0] = 1
    # print(exp_i)

    return exp_i

def replace_with_dict2(ar, dic):
    """
    replace values in ndarray based on dict

    Args:
        ar (_type_): _description_
        dic (_type_): _description_

    Returns:
        _type_: _description_
    """
    k = np.array(list(dic.keys()))
    v = np.array(list(dic.values()))

    # Get argsort indices
    sidx = k.argsort()
    
    ks = k[sidx]
    vs = v[sidx]
    return vs[np.searchsorted(ks,ar)]

def get_i_c_u(csr_matrix, row_delete_baseline=0):
    """
    Calculate complement, intersection and union, ab_ba
    """
    print(f"{datetime.now()}: Reading CSR Matrix...")
    # csr_matrix = sps.load_npz(f"{npz_filepath}")

    # csr_matrix = delete_singleton_labels(csr_matrix, row_delete_baseline)
    # csr_matrix = delete_sample_by_intersection(csr_matrix, row_delete_baseline)
    
    # complement = np.load("Src/Clustering/middle_files/tversky_distance/label_suffixRemoved/relativeCompletement.npy")
    print(f"{datetime.now()}: Convert CSR matrix to dense, and Calculate complement...")
    dense_matrix = csr_matrix.A
    complement = 1 - dense_matrix

    # print(dense_matrix.shape, complement.shape)
    # |(H_A - H_B) + (H_B - H_A)|
    print(f"{datetime.now()}: Calculate relative complement...")
    complement = np.dot(dense_matrix, complement.T)
    complement = complement + complement.T
    print(f"{datetime.now()}: Relative Complement Shape: {complement.shape}")

    print(f"{datetime.now()}: Calculate intersection and union...")
    # intersection = sps.load_npz("Src/Clustering/middle_files/tversky_distance/label_suffixRemoved/intersection.npz")
    intersection = np.dot(csr_matrix, csr_matrix.T)
    # print(np.array_equiv(intersection.A, intersection.A.T))
    union_matrix = intersection + complement

    print(f"union max {np.amax(union_matrix)}, min: {np.amin(union_matrix)}")
    # print(np.array_equiv(union_matrix.A, union_matrix.A.T))

    print(f"{datetime.now()}: union_matrix shape: {union_matrix.shape}")

    # ! -------------- Calculate ab_ba -------------------
    # ! -------------- overlap coefficient -------------------
    # overlap coefficient = $|A \cap B| / min(|A|, |B|)$
    # rows_sum = csr_matrix.getnnz(axis=1)
    # condensed_matrix = [min(pair) for pair in combinations(rows_sum, 2)]
    # pairwise_min_value_matrix = squareform(condensed_matrix)
    # np.fill_diagonal(pairwise_min_value_matrix, rows_sum)

    # overlap_similarity = intersection / pairwise_min_value_matrix
    
    # Hadamard Product
    # modified_jaccard_similarity = np.multiply(i_u, overlap_similarity)
    # modified_jaccard_similarity = np.sqrt(modified_jaccard_similarity)
    # ! ---------------------------------------------------------------

    i_u = intersection / union_matrix
    i_u[np.isnan(i_u)] = 0
    print(np.amax(i_u), np.amin(i_u))
    # i_u[i_u==inf] = 0
    # ! set i/u = 0 if union = 0
    # with np.errstate(divide='ignore'):
    #     # divided = i_u_2 / (1 - i_u_2)
    #     # divided = np.true_divide(i_u, 1-i_u)
    #     i_u = np.true_divide(intersection, union_matrix)

    # i_u[i_u==inf] = 0

    # i_u_2 = np.multiply(i_u, i_u)


    # print(tabulate(
    #     [["Relative Completement", complement.shape, np.array_equiv(complement, complement.T)],
    #      ["I / U", i_u.shape, np.array_equiv(i_u, i_u.T)],
    #      ["(I/U)^2", i_u_2.shape, np.array_equiv(i_u_2, i_u_2.T)]],
    #     headers=["Name", "Shape", "Symmetric"],
    #     tablefmt='fancy_grid'
    # ))
    # print(complement.shape)
    # print(intersection.shape)
    # print(i_u.shape)
    # print(np.array_equiv(i_u, i_u.T))

    return i_u, intersection


def get_ab_ba(csr_matrix) -> np.ndarray:
    """
    在2维的观测向量中, 计算行与行之间的相互差集. 并返回差集较大的值与差集较小的值的比值.
    
    对于第i行和第j行:
    1. i_j 为第i行与第j行的差集, j_i 为第j行与第i行的差集
    2. 返回 `max(|i_j|, |j_i|) / min(|i_j|, |j_i|)`
    3. 如果 min(|i_j|, |j_i|) 为 0, 则直接将比值设为 #! cmath.inf

    Args:
        csr_matrix (CSR Sparse Matrix): 2-D array of observation vectors

    Returns:
        np.ndarray: csr_matrix.shape[0] x csr_matrix.shape[0] 的对称矩阵
    """
    ab_ba = np.zeros(shape=(csr_matrix.shape[0], csr_matrix.shape[0]))
    print(ab_ba.shape)
    for i in range(0, csr_matrix.shape[0]):
        for j in range(0, csr_matrix.shape[0]):
            arr_i = csr_matrix.getrow(i).toarray()[0].astype(bool)
            arr_j = csr_matrix.getrow(j).toarray()[0].astype(bool)
            i_j = (arr_i & ~arr_j).sum()
            j_i = (arr_j & ~arr_i).sum()
            if i_j >= j_i:
                ab_ba[i][j] = j_i / i_j
                if j_i == 0:
                    ab_ba[i][j] = inf
                else:
                    ab_ba[i][j] = i_j / j_i
                    # ab_ba[i][j] = j_i / i_j

            elif i_j < j_i:
                # ab_ba[i][j] = i_j / j_i
                if i_j == 0:
                    ab_ba[i][j] = inf
                else:
                    ab_ba[i][j] = j_i / i_j
                    # ab_ba[i][j] = i_j / j_i

    return ab_ba

def calculate_ab_ba(csr_matrix) -> np.ndarray:
    """
    g(x) = `\frac{2}{1 + e^(-1.0986*x)} - 1`

    Args:
        csr_matrix (_type_): _description_
    """
    print(f"{datetime.now()}: Calculate Sigmoid(|A-B|/|B-A|) ... ...")
    ab_ba = get_ab_ba(csr_matrix)
    i_unique_values = set(np.asarray(ab_ba).ravel())
    # print(i_unique_values)

    sigmoid_i_unique_values = {}
    for i in i_unique_values:
        if i == inf:
            sigmoid_i_unique_values[i] = 1
        else:
            sigmoid_i_unique_values[i] = 2 / (1 + np.exp(-1.0986 * i)) - 1
    
    sigmoid_ab_ba = replace_with_dict2(ab_ba, sigmoid_i_unique_values)
    # print(sigmoid_ab_ba)
    # print(check_symmetric(sigmoid_ab_ba))

    return sigmoid_ab_ba

def calculate_i_min_r(csr_matrix: np.matrix) -> np.ndarray:
    """
    i: intersection
    min_r: min value of complements -> |a-b| and |b-a|
    -------------------------------------------------------
    min(|a-b|, |b-a|) = 0, 有两种状况
    1. a == b --> 可将相似度定为 1
    2. a is a subset of b OR b is a subset of a --> 

    Args:
        csr_matrix (np.matrix): _description_

    Returns:
        np.ndarray: _description_
    """
    intersection = np.dot(csr_matrix, csr_matrix.T)
    min_r = np.zeros(shape=(csr_matrix.shape[0], csr_matrix.shape[0]))
    # print(min_r.shape)
    for i in range(0, csr_matrix.shape[0]):
        for j in range(0, csr_matrix.shape[0]):
            arr_i = csr_matrix.getrow(i).toarray()[0].astype(bool)
            arr_j = csr_matrix.getrow(j).toarray()[0].astype(bool)
            i_j = (arr_i & ~arr_j).sum()
            j_i = (arr_j & ~arr_i).sum()
            min_r[i][j] = min(i_j, j_i)
    
    # i / min_r
    # ! RuntimeWarning: divide by zero encountered in true_divide
    i_min_r = intersection / min_r
    # with np.errstate(divide='ignore'):
    #     i_min_r = np.true_divide(intersection, min_r)

    # i_min_r[i_min_r==inf] = 1

    # apply to function
    # 当 |a| == |b|, 且|I|/|a| = |I|/|b| = 1/2 --> jaccard(a,b) = 1/2, |I| / min(|a|, |b|) = 1 --> g(x) -> (1, 0.5)
    # g(x) = \frac{1}{}

    # scale i_min_r into range(0, 1)
    # first, set inf to 0 in order to get true max value from i_min_r
    i_min_r_copy = np.copy(i_min_r)
    i_min_r_copy[i_min_r_copy==inf] = 0
    mat_max = np.amax(i_min_r_copy)
    mat_min = np.amin(i_min_r_copy)
    print(mat_max, mat_min)

    i_min_r = (i_min_r - mat_min) / (mat_max - mat_min)
    i_min_r[i_min_r==inf] = 1

    return i_min_r


def extract_and_plot(
        keywords_str,
        sim_matrix,
        alpha,
        save_path,
        save_name,
        label_file_path,
        method='complete'):
    """
    Extract data based on family name and plot
    """
    print(f"{datetime.now()}: 1 - Similarity Matrix")
    distance_matrix = np.asmatrix(1 - sim_matrix)
    # distance_matrix = sim_matrix
    print(f"{datetime.now()}: check_symmetric(distance_matrix): {check_symmetric(distance_matrix)}")
    print(f"{datetime.now()}: Max Distance: {np.max(distance_matrix)}")
    print(f"{datetime.now()}: Min Distance: {np.min(distance_matrix)}")
    
    labels = []
    # "Src/Clustering/middle_files/one_hot_encoding_formatData/label_suffixRemoved/vendor_label_hash.txt"
    with open(label_file_path, "r", encoding="utf-8") as f:
        for line in f:
            labels.append(line.strip("\n"))
    
    label_with_index = []
    for l in labels:
        l_index = labels.index(l)
        splitted = re.split("[_ : . /]", l)
        label_with_index.append(
            splitted[0] + "_" + str(l_index) + "_" + splitted[-1].strip("'")
        )


    # labels = [labels[i] for i not in label_index_list]
    # print(len(labels))
    # print(distance_matrix.shape)
    # ------------------ extract labels contain "mirai" ---------------------
    # mirai_index = [labels.index(i) for i in labels if "mirai" in i.lower()]
    # labels = [i for i in labels if "mirai" in i.lower()]
    
    # distance_matrix = distance_matrix[np.ix_(mirai_index, mirai_index)]
    # print(distance_matrix.shape)
    # # -----------------------------------------------------------------------
    
    condensed_distance = squareform(distance_matrix)
    
    # ! plot mapping graph
    # labels_combinations = list(combinations(labels, 2))
    # labels_combinations_distance = {}

    # for i, j in zip(labels_combinations, condensed_distance):
    #     labels_combinations_distance["|".join(i)] = j
        # i.append(j)
    
    # plot_conditoinal_prob_mapping("mirai", 0, "", labels_combinations_distance, alpha)
    # plot_conditoinal_prob_mapping("mirai", 0.8, "", labels_combinations_distance, alpha)
    # ! -----------------------------------------------------------------------------------------

    linkage_matrix = linkage(condensed_distance, method=method, metric="jaccard", optimal_ordering=True)

    print(f"{datetime.now()}: Plot dendrogram")

    fig, ax = plt.subplots(figsize=(80, 60))
    r = dendrogram(
        linkage_matrix,
        # truncate_mode="level",
        # p=200,
        # color_threshold=0.615,
        labels=label_with_index,
        leaf_font_size=5, ax=ax,
        # distance_sort="True",
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

    # print("save figure")

    plt.savefig(f"{save_path}/{save_name}_{keywords_str}_{alpha}_{method}.png", dpi=300)
    # format='pdf', bbox_inches='tight'
    plt.savefig(f"{save_path}/{save_name}_{keywords_str}_{alpha}_{method}.pdf", format='pdf', bbox_inches='tight')

    plt.clf()
    plt.close()

    return r, xtick_labels, colormap, label_with_index, method

def check_symmetric(mat):
    """check if the matrix is symmetric or not

    Args:
        mat (_type_): _description_

    Returns:
        _type_: _description_
    """
    return (mat == mat.T).all()

def jaccard_similarities(mat):
    """
    Referenced from
    http://na-o-ys.github.io/others/2015-11-07-sparse-vector-similarities.html
                                |A & B|  
    Jaccard Distance = 1 - -----------------
                           |A| + |B| - |A & B|

    Args:
        mat (CSR matrix): the mat is binary (0 or 1) matrix and the type is scipy.sparse.csr_matrix.

    Returns:
        _type_: _description_
    """
    # gets the number of non-zeros per row of origin matrix
    # 计算原始矩阵中，每一行中的 1 的个数
    rows_sum = mat.getnnz(axis=1)

    # Use multiplication to get intersection
    # 用矩阵乘法来计算交集
    ab = mat * mat.T

    # mat * mat.T --> mat的行 乘以 mat.T的列
    aa = np.repeat(rows_sum, ab.getnnz(axis=1))

    bb = rows_sum[ab.indices]

    similarities = ab.copy()
    # Hadamard product
    # https://zh.wikipedia.org/wiki/%E9%98%BF%E9%81%94%E7%91%AA%E4%B9%98%E7%A9%8D_(%E7%9F%A9%E9%99%A3)
    # np.reciprocal
    similarities.data /= (aa + bb - ab.data)

    return similarities

def jaccard_distance_csr_sparse_matrix():
    """
    From
    http://na-o-ys.github.io/others/2015-11-07-sparse-vector-similarities.html

    Calculate jaccard similarity for scipy.sparse.csr_matrix
    """
    # Src/Clustering/middle_files/one_hot_encoding_formatData/one_hot_encodingCSR_labels.txt
    # Src/Clustering/middle_files/one_hot_encoding_formatData/label_suffixRemoved/vendor_label_hash.txt
    labels = []
    with open("Src/Clustering/middle_files/one_hot_encoding_formatData/label_suffixRemoved/vendor_label_hash.txt", "r", encoding='utf-8') as f:
        for line in f:
            labels.append(line.strip("\n"))
    # Src/Clustering/middle_files/one_hot_encoding_formatData/one_hot_encodingCSR.npz
    # Src/Clustering/middle_files/one_hot_encoding_formatData/label_suffixRemoved/vendor_label_CSR.npz
    data = scipy.sparse.load_npz("Src/Clustering/middle_files/one_hot_encoding_formatData/label_suffixRemoved/vendor_label_CSR.npz")
    print(data.shape)
    # ! change data type to float64, else
    # ! `TypeError: No loop matching the specified signature and casting was found for ufunc true_divide` will be raised
    data = data.astype("float64")
    sparse_similarities = jaccard_similarities(data)
    dense_similarities = sparse_similarities.todense()

    full_one_matrix = np.full((dense_similarities.shape[0], dense_similarities.shape[1]), 1)
    full_one_matrix = np.asmatrix(full_one_matrix)
    print(full_one_matrix.shape)
    jaccard_distance = np.subtract(full_one_matrix, dense_similarities)
    print(jaccard_distance.shape)

    condensed_distance = squareform(jaccard_distance)
    print((condensed_distance))
    
    # save jaccard_distance to pandas dataframe
    jaccard_distance_df = pd.DataFrame(jaccard_distance, index=labels, columns=labels)
    print(jaccard_distance_df)
    jaccard_distance_df.to_pickle("Src/Clustering/middle_files/original_jaccard_distance/label_suffixRemoved/jaccard_distance_dense.pkl")

    # save condensed_distance to pandas dataframe
    np.save("Src/Clustering/middle_files/original_jaccard_distance/label_suffixRemoved/jaccard_distance_condensed.npy", condensed_distance)

def combination_intersection(json_filepath: str, save_path: str):
    """
    get pairwise intersection
    https://stackoverflow.com/questions/27369373/pairwise-set-intersection-in-python/27370005#27370005
    """
    keywords_str = os.path.basename(json_filepath).split(".")[0]
    with open(f"{json_filepath}", "r", encoding='utf-8') as json_obj:
        data = json.load(json_obj)
    hashes = list(data.values())
    hashes = [set(i) for i in hashes]
    labels = list(data.keys())

    idx = range(len(labels))
    pairs = combinations(idx, 2)
    nt = lambda a, b: hashes[a].intersection(hashes[b])

    res = dict([
        (t, list(nt(*t))) for t in pairs
    ])
    label_intersection = {}
    for pair, overlap in res.items():
        label_intersection[labels[pair[0]] + "|" + labels[pair[1]]] = overlap
    # res1 = {t: nt(*t) for t in pairs}
    # print(res1)
    with open(f"{save_path}/{keywords_str}_pairwise_intersection.json", "w", encoding='utf-8') as f:
        json.dump(label_intersection, f)

def inclusion_relationship(set0: set, set1: set):
    """ inclusion relationship or not

    Args:
        set0 (set): _description_
        set1 (set): _description_
    """
    if set0.issubset(set1) or set1.issubset(set0):
        return "O"
    else:
        return "X"


def plot_pairwise_intersection(
            pairwise_intersection_json_filepath: str,
            label_hashes_json_filepath: str,
            labels: list,
            colormap: dict,
            save_path: str
            ):
    """_summary_

    Args:
        pairwise_intersection_json_filepath (str): _description_
        label_hashes_json_filepath (str): _description_
        labels (list): _description_
        colormap (dict): _description_
        save_path (str): _description_
    """
    print(f"{datetime.now()}: Pairwise intersection")
    keywords_str = os.path.basename(label_hashes_json_filepath).split(".")[0]

    with open(f"{pairwise_intersection_json_filepath}", "r", encoding='utf-8') as f:
        report = json.load(f)
    # read label-hashes json file
    with open(f"{label_hashes_json_filepath}", "r", encoding='utf-8') as json_obj:
        data = json.load(json_obj)
    # labels = list(data.keys())

    intersection_df = pd.DataFrame(np.random.randint(0, 100, size=(len(labels), len(labels))), columns=labels, index=labels)
    pairwise_size_df = pd.DataFrame(np.random.randint(0, 100, size=(len(labels), len(labels))), columns=labels, index=labels)

    # print(intersection_df)
    for pair in report:
        l0 = pair.split("|")[0]
        l1 = pair.split("|")[1]
        s0 = len(data[l0])
        s1 = len(data[l1])
        set0 = set(data[l0])
        set1 = set(data[l1])

        intersection_df.at[l0, l1] = len(report[pair])
        intersection_df.at[l1, l0] = len(report[pair])
        intersection_df.at[l0, l0] = s0
        intersection_df.at[l1, l1] = s1

        pairwise_size_df.at[l0, l1] = "(" + str(s0) + "," + str(s1) + "," + str(len(report[pair])) + "," + inclusion_relationship(set0, set1) + ")"
        pairwise_size_df.at[l1, l0] = "(" + str(s1) + "," + str(s0) + "," + str(len(report[pair])) + "," + inclusion_relationship(set0, set1) + ")"
        pairwise_size_df.at[l0, l0] = "(" + str(s0) + "," + str(s0) + "," + str(s0) + ")"
        pairwise_size_df.at[l1, l1] = "(" + str(s1) + "," + str(s1) + "," + str(s1) + ")"


    # print(intersection_df)
    # print(pairwise_size_df)

    print(f"{datetime.now()}: Plotting")

    mask = np.triu(np.ones_like(intersection_df.values, dtype=bool))

    f, ax = plt.subplots(figsize=(120, 100))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(intersection_df.values,
                annot=pairwise_size_df.values,
                fmt='',
                mask=mask,
                annot_kws={"fontsize": 3},
                cmap=cmap,
                vmax=.3,
                center=0,
                square=True, cbar=False, linewidths=.9, ax=ax)
    
    plt.xticks(ticks=range(0, len(labels)), labels=labels, rotation=90, fontsize=10)
    plt.yticks(ticks=range(0, len(labels)), labels=labels, fontsize=10)
    # ax.set_xlim([0, len(labels)])
    # ax.set_xticks(range(0, len(labels)))
    # ax.set_xticklabels(labels, fontsize=10, rotation=90)

    # ax.set_ylim([0, len(labels)])
    # ax.set_yticks(range(0, len(labels)))
    # ax.set_yticklabels(labels, fontsize=10)

    # ax2 = ax.twiny()
    # ax2.set_xlim([0, ax.get_xlim()[1]])
    # ax2.set_xticks(ax.get_xticks())
    # ax2.set_xticklabels(labels, fontsize=10, rotation=-90)
    # plt.subplots_adjust(left=0.1)
    # ax.set_xticks(np.arange(0, 1102, 1))
    # ax.set_xticklabels(labels)
    # ax.set_yticklabels(labels)

    # set tick color
    # for l, c in zip(plt.gca().get_xticklabels(), leaf_color):
    #     l.set_color(c)
    # for l, c in zip(plt.gca().get_yticklabels(), leaf_color):
    #     l.set_color(c)
    
    for label in ax.xaxis.get_ticklabels():
        label_text = label.get_text()
        keys = [i for i in list(colormap.keys()) if i in label_text]
        label_index = ax.xaxis.get_ticklabels().index(label)
        if len(keys) > 0:
            ax.xaxis.get_ticklabels()[label_index].set_color(colormap[keys[0]])
        else:
            ax.xaxis.get_ticklabels()[label_index].set_color("pink")

    for label in ax.yaxis.get_ticklabels():
        label_text = label.get_text()
        keys = [i for i in list(colormap.keys()) if i in label_text]
        label_index = ax.yaxis.get_ticklabels().index(label)
        if len(keys) > 0:
            ax.yaxis.get_ticklabels()[label_index].set_color(colormap[keys[0]])
        else:
            ax.yaxis.get_ticklabels()[label_index].set_color("pink")
    
    plt.tight_layout()
    print(f"{datetime.now()}: Saving...")
    # plt.savefig(f"{save_path}/{keywords_str}_pairwise_intersection_size.png", dpi=400)
    plt.savefig(f"{save_path}/{keywords_str}_pairwise_intersection_size.pdf", format='pdf', bbox_inches='tight')
    plt.clf()
    plt.close()

def plot_pairwise_intersection_based_matrix(
            csr_matrix,
            intersection_csr_matrix,
            labels: list,
            xtick_labels: list,
            colormap: dict,
            save_path: str,
            keywords_str: str,
            alpha: int,
            method: str):
    """_summary_

    Args:
        csr_matrix (_type_): _description_
        intersection_csr_matrix (_type_): _description_
        labels (list): _description_
    """
    print(f"{datetime.now()}: Plot Pairwise intersecton...")
    intersection_arr = intersection_csr_matrix.A
    intersection_df = pd.DataFrame(intersection_arr, columns=labels, index=labels)
    print(intersection_arr)
    print(intersection_df)

    pairwise_size_df = pd.DataFrame(np.random.randint(0, 100, size=(len(labels), len(labels))), columns=xtick_labels, index=xtick_labels)
    
    for pair in combinations(labels, 2):
        # print(pair)
        row_0 = csr_matrix[labels.index(pair[0]), :].A.astype(bool)
        row_1 = csr_matrix[labels.index(pair[1]), :].A.astype(bool)
        count_0 = row_0.sum()
        count_1 = row_1.sum()

        union = (row_0 | row_1).sum()
        relative_complement_01 = (row_0 > row_1).sum()
        relative_complement_10 = (row_1 > row_0).sum()
        # print(f"{pair[0]} {pair[1]}: {count_0} {count_1} {union} {relative_complement_01} {relative_complement_10}")

        # print(pair, count_0, count_1, intersection_df.loc[pair[0], pair[1]], )
        intersection_vec = row_0 & row_1
        if np.array_equal(intersection_vec, row_0) or np.array_equal(intersection_vec, row_1):
            containing_relationship = "O"
        else:
            containing_relationship = "X"
        # print(str(count_0) + "," + str(count_1) + "," + str(intersection_df.loc[pair[0], pair[1]]) + "," + containing_relationship \
        #     + "\n" + str(relative_complement_01) + "," + str(relative_complement_10) + "," + str(union))
        
        pairwise_size_df.at[pair[0], pair[1]] = str(count_0) + "," + str(count_1) + "," + str(intersection_df.loc[pair[0], pair[1]]) + "," + containing_relationship \
            + "\n" + str(relative_complement_01) + "," + str(relative_complement_10) + "," + str(union)
        
        pairwise_size_df.at[pair[1], pair[0]] = str(count_1) + "," + str(count_0) + "," + str(intersection_df.loc[pair[0], pair[1]]) + "," + containing_relationship \
            + "\n" + str(relative_complement_10) + "," + str(relative_complement_01) + "," + str(union)
        pairwise_size_df.at[pair[0], pair[0]] = str(count_0) + "," + str(count_0) + "," + str(intersection_df.loc[pair[0], pair[0]]) + ",O"
        pairwise_size_df.at[pair[1], pair[1]] = str(count_1) + "," + str(count_1) + "," + str(intersection_df.loc[pair[1], pair[1]]) + ",O"

        # print(pairwise_size_df.loc[pair[0], pair[1]])
    
    print(pairwise_size_df)

    print(f"{datetime.now()}: Plotting")
    
    intersection_df = intersection_df.reindex(index=xtick_labels, columns=xtick_labels)
    mask = np.triu(np.ones_like(intersection_df.values, dtype=bool))


    f, ax = plt.subplots(figsize=(120, 100))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(intersection_df.values,
                annot=pairwise_size_df.values,
                fmt='',
                mask=mask,
                annot_kws={"fontsize": 3},
                cmap=cmap,
                vmax=.3,
                center=0,
                square=True, cbar=False, linewidths=.9, ax=ax)
    
    plt.xticks(ticks=range(0, len(labels)), labels=xtick_labels, rotation=90, fontsize=10)
    plt.yticks(ticks=range(0, len(labels)), labels=xtick_labels, fontsize=10)

    for label in ax.xaxis.get_ticklabels():
        label_text = label.get_text()
        keys = [i for i in list(colormap.keys()) if i in label_text]
        label_index = ax.xaxis.get_ticklabels().index(label)
        if len(keys) > 0:
            ax.xaxis.get_ticklabels()[label_index].set_color(colormap[keys[0]])
        else:
            ax.xaxis.get_ticklabels()[label_index].set_color("pink")

    for label in ax.yaxis.get_ticklabels():
        label_text = label.get_text()
        keys = [i for i in list(colormap.keys()) if i in label_text]
        label_index = ax.yaxis.get_ticklabels().index(label)
        if len(keys) > 0:
            ax.yaxis.get_ticklabels()[label_index].set_color(colormap[keys[0]])
        else:
            ax.yaxis.get_ticklabels()[label_index].set_color("pink")
    
    plt.tight_layout()
    print(f"{datetime.now()}: Saving...")
    # plt.savefig(f"{save_path}/{keywords_str}_pairwise_intersection_size.png", dpi=400)
    plt.savefig(f"{save_path}/{keywords_str}_{str(alpha)}_{method}_pairwise_intersection_size.pdf", format='pdf', bbox_inches='tight')
    plt.clf()
    plt.close()


def delete_singleton_labels(csr_matrix, n):
    """_summary_

    Args:
        csr_matrix (_type_): _description_
    """
    mat = csr_matrix.A

    # results_index = []
    col_sum_list = mat.sum(axis=0).tolist()
    # row_sum_list = mat.sum(axis=1).tolist()
    # row_index_list = [i for i, x in enumerate(row_sum_list) if x <= n]
    # for i in row_index_list:
    #     row = mat[i, :]
    #     row_nonzero_index = [i for i, x in enumerate(row) if x == 1]
    #     col_mat = mat[row_nonzero_index, :]
    #     if col_mat.sum() == n:
    #         results_index.append(i)

    # print(len(col_sum_list))
    col_index_list = [i for i, x in enumerate(col_sum_list) if x <= n]
    print(len(col_index_list))
    mat_del = np.delete(mat, col_index_list, axis=1)
    # # print(len(index_list))
    # row_index = []
    # for i in col_index_list:
    #     col = mat[:, i]
    #     # print(len(col), list(col).count(1))
    #     # print(list(col).count(1))
    #     nonzero_index = [i for i, x in enumerate(col) if x == 1]
    #     # print(nonzero_index)
    #     row_index.extend(nonzero_index)
    # row_index = list(set(row_index))
    # # print(row_index)
    # print(len(row_index))
    # results_index = [int(i) for i in results_index]
    
    # mat_del = np.delete(mat, results_index, axis=0)
    # # print(mat.shape, len(index_list), mat_del.shape)
    print(mat_del)
    csr_matrix_del = sps.csr_matrix(mat_del)
    # print(csr_matrix_del.shape)
    return csr_matrix_del

    # return csr_matrix_del, results_index

def delete_sample_by_intersection(csr_matrix, n):
    """
    当两个集合的交集大小只有 n 个时, 删除这 n 个样本

    Args:
        csr_matrix (_type_): _description_
        n (_type_): _description_
    """
    # original_shape = 
    intersection = np.dot(csr_matrix, csr_matrix.T)
    n_intersection = intersection==n
    row_col = n_intersection.nonzero()
    print(len(row_col[0]))
    sample_index = []
    for row, col in zip(row_col[0], row_col[1]):
        selected_rows = csr_matrix[[row, col], :].A.astype(bool)
        and_operation = selected_rows.all(axis=0)
        sample_index.extend(list(np.where(and_operation)[0]))
        # print()
        # print(selected_rows.nonzero())
        # print(row, col)
    # mat = csr_matrix.A.astype(bool)
    # print(mat)
    sample_index = list(set(sample_index))
    print(len(sample_index))
    print(csr_matrix.shape)
    
    keep = ~np.in1d(np.arange(csr_matrix.shape[1]), sample_index, assume_unique=True)
    csr_matrix_del = csr_matrix[:, np.where(keep)[0]]
    # print(csr_matrix_del.shape)

    return csr_matrix_del

def logistic_function(
        i_u: np.matrix,
        alpha: int):
    """
    `f(x) = 1 / (1 + e^(-\alpha * (i_u - 0.5)))`

    Args:
        i_u (np.matrix): _description_
        alpha (int): _description_
    """
    i_unique_values = set(np.asarray(i_u).ravel())
    exp_i_unique_values = {}
    for i in i_unique_values:
        # ! i_u 为 1 时, 相似度应该为 1, 但是 np.exp(- alpha * (i - 0.5)) 在 i=1 时，相似度无法取到 1.
        if i == 1:
            exp_i_unique_values[i] = 0
        else:
            exp_i_unique_values[i] = np.exp(- alpha * (i - 0.5))
    
    exp_i_u = replace_with_dict2(i_u, exp_i_unique_values)

    jaccard_sim = 1 / (1 + exp_i_u)
    # ! i_u 为 0 时, 相似度应该为 0, 但是 np.exp(- alpha * (i - 0.5)) 在 i=0 时，无法取到 0.
    zero_limit = 1 / (1 + np.exp(- alpha * (0 - 0.5))) # 006734634426505237
    jaccard_sim[jaccard_sim == zero_limit] = 0

    # print(jaccard_sim)
    print(f"{datetime.now()}: Max: {np.max(jaccard_sim)}")
    np.fill_diagonal(jaccard_sim, 1)
    # similarity_matrix[similarity_matrix == 0.5] = 1
    print(f"{datetime.now()}: similarity_matrix: {check_symmetric(jaccard_sim)}")

    return jaccard_sim


def s_shaped(
        i_u: np.matrix,
        alpha: int):
    """
    `f(x) = 1 - \franc{1}{1 + (\frac{x}{1-x})^{\alpha}}}`

    Args:
        i_u (np.matrix): _description_
        alpha (int): _description_
    """
    with np.errstate(divide='ignore'):
        # divided = i_u_2 / (1 - i_u_2)
        # divided = np.true_divide(i_u, 1-i_u)
        divided = np.true_divide(i_u, 1-i_u)

    divided[divided==inf] = 1

    # print(divided)
    # print("divided: ", check_symmetric(divided))
    # print(np.linalg.det(divided))
    # ! divided 为奇异矩阵，无法求逆

    a = np.power(divided, alpha)
    # a = fractional_matrix_power(divided, alpha)
    # print("a: ", check_symmetric(a))

    # # ! numpy.linalg.LinAlgError: Singular matrix
    jaccard_sim = 1 - 1 / (1 + a)
    # print(similarity_matrix)
    np.fill_diagonal(jaccard_sim, 1)
    jaccard_sim[jaccard_sim == 0.5] = 1
    print(f"{datetime.now()}: similarity_matrix: {check_symmetric(jaccard_sim)}")

    return jaccard_sim


def polynomials_function(mat, alpha):
    """
    f(x) = 1 - (1 - x)^a

    Args:
        mat (_type_): _description_
        alpha (_type_): _description_
    """

    return 1 - np.power((1 - mat), alpha)


# 
def containing_difference_coefficient(csr_matrix, alpha, beta):
    """_summary_


    Args:
        csr_matrix (_type_): _description_
    """
    intersection = np.dot(csr_matrix, csr_matrix.T)
    max_r = np.zeros(shape=(csr_matrix.shape[0], csr_matrix.shape[0]))

    ab_u = np.zeros(shape=(csr_matrix.shape[0], csr_matrix.shape[0]))

    # print(min_r.shape)
    for i in range(0, csr_matrix.shape[0]):
        for j in range(0, csr_matrix.shape[0]):
            arr_i = csr_matrix.getrow(i).toarray()[0].astype(bool)
            arr_j = csr_matrix.getrow(j).toarray()[0].astype(bool)
            i_j = (arr_i & ~arr_j).sum()
            j_i = (arr_j & ~arr_i).sum()
            max_r[i][j] = max(i_j, j_i)

            union = (arr_i | arr_j).sum()
            ab_u[i][j] = (i_j + j_i) / union
            # ab_u[i][j] = union / (i_j + j_i)


    # -------------------------------------------------------
    #                       i_max_r: I / max(|a-b|, |b-a|)
    #           g(y) = 1 / (1 + e^(-\alpha * (y - 0.5) ))
    # -------------------------------------------------------
    i_max_r = intersection / max_r
    # i_max_r[i_max_r==inf] = 1

    # apply to g(y)
    unique_values = set(np.asarray(i_max_r).ravel())
    exp_unique_values = {}
    for i in unique_values:
        if i == inf:
            exp_unique_values[i] = 1
        elif i == 0:
            exp_unique_values[i] = 0
        else:
            exp_unique_values[i] = 1 / (1 + np.exp(- alpha * (i - 0.5)))
    
    i_max_r_coefficient = replace_with_dict2(i_max_r, exp_unique_values)

    # i_max_r_coefficient = 1 / (1 + exp_i_max_r)

    # ------------------------------------------------
    #               ab_u: (|a-b| + |b-a|) / u
    #       h(z) = 2 - 2 / (1 + e^(-beta * z)) ✖
    #       h(z) = 1 / (1 + e^(-beta * (z - 2)))　　✖
    #       h(z) = 1 - z^beta
    # ------------------------------------------------
    ab_u_coefficient = 1 - np.power(ab_u, beta)
    # ab_u_unique_values = set(np.asarray(ab_u).ravel())
    # exp_ab_u_unique_values = {}
    # for i in ab_u_unique_values:
    #     if i == 0:
    #         exp_ab_u_unique_values[i] = 1
    #     elif i == 1:
    #         exp_ab_u_unique_values[i] = 0
    #     else:
    #         # exp_ab_u_unique_values[i] = 2 - (2 / ( 1 + np.exp(- beta * i)))
    #         exp_ab_u_unique_values[i] = 1 / ( 1 + np.exp(- beta * (i-2)))


    # ab_u_coefficient = replace_with_dict2(ab_u, exp_ab_u_unique_values)

    # ab_u_coefficient = 2 - (2 / (1 + exp_ab_u))

    # -----------------------------------------------
    print(f"i_max_r: {np.amax(i_max_r_coefficient)}, {np.amin(i_max_r_coefficient)}")
    print(f"ab_u: {np.amax(ab_u_coefficient)}, {np.amin(ab_u_coefficient)}")
    coefficient = np.multiply(i_max_r_coefficient, ab_u_coefficient)
    print(f"coefficient: {np.amax(coefficient)}, {np.amin(coefficient)}")

    # return i_max_r_coefficient, ab_u_coefficient
    return coefficient

# (|a-b| + |b-a|) / u

def scale_mat2_01(mat):

    """

    Args:
        mat (_type_): symmetric similarity matrix
    """
    mat_copy = np.copy(mat)
    max_value = np.amax(mat_copy)
    min_value = np.min(mat_copy)

    return (mat - min_value) / (max_value - min_value)

def sqrt_abba_u(csr_matrix, alpha_logistic, alpha_s_shaed, beta):
    """
    |A| >= |B|
    x = |I| / |A|
    f(x) = 1 / (1 + e^(-alpha*(x-0.5)))

    y = sqrt(|A-B|*|B-A|) / (|U|/2)
    g(y) = 1 - y^beta

    Args:
        csr_matrix (_type_): _description_
        alpha (_type_): _description_
    """
    intersection = np.dot(csr_matrix, csr_matrix.T)

    sqrt_abba_u_mat = np.zeros(shape=(csr_matrix.shape[0], csr_matrix.shape[0]))

    # |A| >= |B|
    max_a = np.zeros(shape=(csr_matrix.shape[0], csr_matrix.shape[0]))

    # print(min_r.shape)
    for i in range(0, csr_matrix.shape[0]):
        for j in range(0, csr_matrix.shape[0]):
            arr_i = csr_matrix.getrow(i).toarray()[0].astype(bool)
            arr_j = csr_matrix.getrow(j).toarray()[0].astype(bool)
            i_j = (arr_i & ~arr_j).sum()
            j_i = (arr_j & ~arr_i).sum()
            union = (arr_i | arr_j).sum()

            # x = sqrt(|A-B|*|B-A|) / (|U|/2) = 2 * sqrt(|A-B|*|B-A|) / |U|
            sqrt_abba_u_mat[i][j] = 2 * np.sqrt(i_j * j_i) / union
            max_a[i][j] = max(arr_i.sum(), arr_j.sum())
    
    # max_a = np.asmatrix(max_a)
    i_max_a = intersection / max_a

    logisti_i_max_a = logistic_function(
        i_max_a,
        alpha_logistic
    )

    s_shaped_i_max_a = s_shaped(
        i_max_a,
        alpha_s_shaed
    )
    
    sqrt_abba_u_results = 1 - np.power(sqrt_abba_u_mat, beta)

    return intersection, np.multiply(logisti_i_max_a, sqrt_abba_u_results), np.multiply(s_shaped_i_max_a, sqrt_abba_u_results)


def i_min_ab_a(csr_matrix: sps.csr_matrix, logistic_alpha, s_shaped_alpha, beta):
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
        csr_matrix (np.matrix): _description_
    """
    intersection = np.dot(csr_matrix, csr_matrix.T)
    min_b = np.zeros(shape=(csr_matrix.shape[0], csr_matrix.shape[0]))
    ab_a_ba_b = np.zeros(shape=(csr_matrix.shape[0], csr_matrix.shape[0]))

    # print(min_r.shape)
    for i in range(0, csr_matrix.shape[0]):
        for j in range(0, csr_matrix.shape[0]):
            arr_i = csr_matrix.getrow(i).toarray()[0].astype(bool)
            arr_j = csr_matrix.getrow(j).toarray()[0].astype(bool)
            arr_i_count = arr_i.sum()
            arr_j_count = arr_j.sum()
            i_j = (arr_i & ~arr_j).sum()
            j_i = (arr_j & ~arr_i).sum()
            # union = (arr_i | arr_j).sum()

            min_b[i][j] = min(arr_i.sum(), arr_j.sum())
            ab_a_ba_b[i][j] = (i_j /arr_i_count + j_i / arr_j_count) / 2
    
    i_min_b = intersection / min_b
    # `f(x) = 1/(1+e^(-\alpha * (x-0.5)))`
    logistic_i_min_b = logistic_function(
        i_min_b,
        logistic_alpha
    )
    s_shaped_i_min_b = s_shaped(
        i_min_b,
        s_shaped_alpha
    )

    # `g(y) = 1 - y^beta`
    g_ab_a_ba_b = 1 - np.power(ab_a_ba_b, beta)

    logisti_similarity = np.multiply(logistic_i_min_b, g_ab_a_ba_b)
    s_shaped_similarity = np.multiply(s_shaped_i_min_b, g_ab_a_ba_b)

    return intersection, logisti_similarity, s_shaped_similarity








    