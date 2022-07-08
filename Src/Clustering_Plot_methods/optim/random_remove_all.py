import sys
from datetime import datetime
sys.path.append("/home/syuu/Project/nict_clustering")
import pyximport
pyximport.install()
from hcluster_c import set_dist_clustering
from Clustering_Plot_methods.functions import plot_dendrogram


if __name__ == "__main__":
    start = datetime.now()

    random_all_npz = "/home/syuu/Project/nict_clustering/latest_analysis_date/clustering/20220620/input_data/random_remove/20220620_all_CSR.npz"
    random_all_label = "/home/syuu/Project/nict_clustering/latest_analysis_date/clustering/20220620/input_data/random_remove/20220620_all_labels.txt"

    set_dist_clustering(
        npz_filepath=random_all_npz,
        label_file_path=random_all_label,
        linkage_matrix_save_path="/home/syuu/Project/nict_clustering/hcluster/files/linkage_matrix/random",
        save_name="20220620_random_part"
    )
    # plot_dendrogram(
    #     save_path="/home/syuu/Project/nict_clustering/hcluster/files/logistic/random_remove",
    #     label_file_path=random_all_label,
    #     npz_filepath=random_all_npz,
    #     linkage_matrix_filepath='/home/syuu/Project/nict_clustering/hcluster/files/linkage_matrix/random/20220620_random_part.npy',
    #     save_name="20220620_gcsht",
    #     family_name_count_json_filepath=["gafgyt", "coinminer", "oinminer", 'sabsik', 'hajime', 'tsunami', 'mirai']
    # )

