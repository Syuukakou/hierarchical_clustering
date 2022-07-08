from typing import Counter
import matplotlib.pyplot as plt
import json
import sys
from collections import Counter
import matplotlib as mpl
sys.path.append("/home/syuu/Project/nict_clustering")

def plot():
    """
    plot func
    """
    with open("latest_analysis_date/Files/tsunami_hajime.json", "r", encoding='utf-8') as f:
        data = json.load(f)
    label_hash_counts = {}
    for label, hashes in data.items():
        label_hash_counts[label] = len(hashes)
    # print(label_hash_counts)
    label_hash_counts = dict(
                            sorted(
                                dict(Counter(label_hash_counts)).items(),
                                key=lambda item: item[1],
                                reverse=True
                                )
                            )
    
    labels = list(label_hash_counts.keys())
    counts = list(label_hash_counts.values())

    fig, ax = plt.subplots(figsize=(30, 10))
    bar_list = ax.bar(labels, counts)
    plt.xticks(fontsize=5, rotation=90)
    plt.draw()
    # x_tick_labels = (ax.get_xticklabels())
    # print(x_tick_labels)
    for label in ax.xaxis.get_ticklabels():
        label_text = label.get_text()
        if "hajime" in label_text:
            label_index = ax.xaxis.get_ticklabels().index(label)
            ax.xaxis.get_ticklabels()[label_index].set_color("red")
            bar_list[label_index].set_color("red")
    # for i in range(len(x_tick_labels)):
    #     if "hajime" in x_tick_labels[i].get_text():
    #         ax.get_xticklabels()[i].set_color("red")
    ax.margins(x=0)
    plt.tight_layout()
    plt.savefig("latest_analysis_date/Files/tsunami_hajime_statistics.png", dpi=300)

if __name__ == "__main__":
    plot()