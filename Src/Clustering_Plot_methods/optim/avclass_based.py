import re
import sys
from datetime import datetime
sys.path.append("/home/syuu/Project/nict_clustering")
import pyximport
pyximport.install()
from hcluster_c import set_dist_clustering
from Clustering_Plot_methods.functions import plot_dendrogram



# 34, 85, 93, 119, 260, 261, 197, 327, 221, 152, 103, 242, 308, 37, 177, 225

if __name__ == "__main__":
    start = datetime.now()

    all_npz_file = "/home/syuu/Project/nict_clustering/latest_analysis_date/clustering/20220620/input_data/avclass_based/20220620_CSR.npz"
    all_label_text_file = "/home/syuu/Project/nict_clustering/latest_analysis_date/clustering/20220620/input_data/avclass_based/20220620_labels.txt"
    all_family_name_count_json = "/home/syuu/Project/nict_clustering/Data_preprocess/data/avclass_based/family_name_counter_20220620.json"
    
    gcsht_npz = "/home/syuu/Project/nict_clustering/latest_analysis_date/clustering/20220620/input_data/avclass_based/20220620_gcsht_CSR.npz"
    gcsht_label = "/home/syuu/Project/nict_clustering/latest_analysis_date/clustering/20220620/input_data/avclass_based/20220620_gcsht_labels.txt"

    # set_dist_clustering(
    #     npz_filepath=gcsht_npz,
    #     label_file_path=gcsht_label,
    #     linkage_matrix_save_path="/home/syuu/Project/nict_clustering/hcluster/files/linkage_matrix/avclass",
    #     save_name="20220620_gcsht"
    # )
    # ["gafgyt", "coinminer", "oinminer", 'sabsik', 'hajime', 'tsunami', 'mirai']
    plot_all_args = {
        "save_path": "/home/syuu/Project/nict_clustering/hcluster/files/logistic/avclass_based",
        "label_file_path": all_label_text_file,
        "npz_filepath": all_npz_file,
        "linkage_matrix_filepath": "hcluster/files/linkage_matrix/avclass/20220620_all.npy",
        "save_name": "20220620_all",
        "family_name_count_json_filepath": all_family_name_count_json,
        "plot_pairwise_intersection": False,
        "pairwise_keyword_str": "all"
    }
    plot_part_args = {
        "save_path": "/home/syuu/Project/nict_clustering/hcluster/files/logistic/avclass_based",
        "label_file_path": gcsht_label,
        "npz_filepath": gcsht_npz,
        "linkage_matrix_filepath": "hcluster/files/linkage_matrix/avclass/20220620_gcsht.npy",
        "save_name": "20220620_part",
        "family_name_count_json_filepath": ["gafgyt", "coinminer", 'sabsik', 'hajime', 'tsunami', 'mirai'],
        "plot_pairwise_intersection": False,
        "pairwise_keyword_str": "gcsht"
    }
    plot_dendrogram(**plot_all_args)
    plot_dendrogram(**plot_part_args)