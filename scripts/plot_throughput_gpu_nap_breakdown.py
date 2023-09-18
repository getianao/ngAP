import sys
import os
import pandas as pd
import numpy as np
import glob
import seaborn as sns
import figurePlotter
from dict_config import *

configs_dict = {
    # sota
    "before-infant_": ["iNFAnt", -1],
    "before-nfacg_": ["NFA-CG", 2],
    "before-newtran-nt_": ["NT", -3],
    "before-newtran-ntmac_": ["NT-MaC", -4],
    "before-hotstarttt_": ["HotStartTT", -5],
    "before-hotstart-nt_": ["GPU-NFA", 10],
    "before-hotstart-ntmac_": ["HotStart-Mac", -7],
    "before-hyperscan_": ["HyperScan", -8],
    "before-runahead-cc4_": ["AsyncAP", 9],
    # NAP
    "o0-blocking_": ["O0Blocking", -50],
    "o0-nonblocking-NAP_": ["O0NAP", -51],
    "o1-nonblocking_": ["O1", -52],
    "o1-nonblocking-aas_": ["O1aas", -53],
    "o1-nonblocking-unique_": ["O1unique", -54],
    "o4-nonblocking-r1_": ["O4r1", -55],
    "o4-nonblocking-r1f_": ["O4r1f", -56],
    "o4-nonblocking-r2_": ["O4r2", -56],
    "o3-nonblocking-p1_": ["O3p1", -57],
    "o3-nonblocking-p2_": ["O3p2", -58],
    "o3-nonblocking-p3_": ["O3p3", -59],
    "oa-nonblocking-all-p2r1_": ["OAp2r1", -60],
    "oa-nonblocking-all-p2r1f_": ["OAp2r1f", -61],
    "oa-nonblocking-all-p3r1_": ["OAp3r1", -62],
    "oa-nonblocking-all-p3r1f_": ["OAp3r1f", -63],
    # "oa-nonblocking-default-32-best": ["ngAP-default-32", 63.1],
    # "oa-nonblocking-default-128-best": ["ngAP-default-128", 63.2],
    "oa-nonblocking-default-256-best": ["ngAP-default-256", 63.3],
    "oa-nonblocking-default-best": ["ngAP-default", 63.5],
    "oa-nonblocking-all-best": ["ngAP-Best", 64],
    "o0-blocking-breakdown_": ["BAP", 81],
    "o0-nonblocking-NAP-breakdown_": ["ngAP", 82],
    "o1-nonblocking-breakdown_": ["ngAP+$\mathregular{O^1}$", 83],
    # "o4-nonblocking-r-breakdown_": ["NAP+O3", -84],
    "o3-nonblocking-p-breakdown_": ["ngAP+$\mathregular{O^2}$", 85],
    "oa-nonblocking-all-breakdown_": ["ngAP+$\mathregular{O^3}$", 86],
}

# configs_groups = [["o0-blocking_"], ["o0-nonblocking-NAP_"], ["o1-nonblocking_"], ["o4-nonblocking-r1_", "o4-nonblocking-r1f_", "o4-nonblocking-r2_", "o4-nonblocking-r2f_"],
#                   ["o3-nonblocking-p1_", "o3-nonblocking-p2_", "o3-nonblocking-p3_"], ["oa-nonblocking-all-p2r1_", "oa-nonblocking-all-p2r1f_", "oa-nonblocking-all-p3r1_", "oa-nonblocking-all-p3r1f_"]]
configs_groups = [
    ["o0-blocking_"],
    ["o0-nonblocking-NAP_"],
    ["o1-nonblocking_"],
    [
        "o4-nonblocking-r1_",
        "o4-nonblocking-r1f_",
        "o4-nonblocking-r2_",
        "o4-nonblocking-r2f_",
    ],
    ["o3-nonblocking-p3_"],
    ["oa-nonblocking-all-p3r1f_", "oa-nonblocking-all-p3r1_"],
]

configs_groups_names = [
    "o0-blocking-breakdown_",
    "o0-nonblocking-NAP-breakdown_",
    "o1-nonblocking-breakdown_",
    "o4-nonblocking-r-breakdown_",
    "o3-nonblocking-p-breakdown_",
    "oa-nonblocking-all-breakdown_",
]


def normalize_data(data, normalize_to_column_name):
    row_names = data.index.tolist()
    # error_value = 0
    for row_name in row_names:
        normalize_to = data.loc[row_name, normalize_to_column_name]
        if normalize_to <= 0:
            data.loc[row_name] = np.nan
        else:
            data.loc[row_name] = data.loc[row_name] / normalize_to
        data.loc[row_name][data.loc[row_name] < 0] = np.nan
    return data


def remove_nan(data, value):
    row_names = data.index.tolist()
    # error_value = 0
    for row_name in row_names:
        data.loc[row_name][data.loc[row_name].isna()] = value
    return data


def save_to_csv(data, csv_path):
    csv_file = os.path.splitext(os.path.abspath(csv_path))[0] + ".csv"
    print("Save data to", csv_file)
    data.to_csv(csv_file)


def geo_mean(x):
    a = np.log(x)
    return np.exp(a.mean())


def plot(path_list, figurePath, ylabel):
    # Load data
    data = pd.DataFrame()
    for path in path_list:
        print(path)
        data_apps = pd.DataFrame()
        csv_files = glob.glob(os.path.abspath(path) + "/*.{}".format("csv"))
        for file in csv_files:
            df = pd.read_csv(file)
            if df.empty:
                continue
            df = df.T
            df.columns = df.loc["config"]
            df = df.drop("config", axis=0)
            df["App"] = df.index.tolist()
            data_apps = pd.concat([data_apps, df])
            # print(data_apps, '\n')
        if data.empty:
            data = data_apps
        else:
            data = data.merge(data_apps, how="outer", on="App")

    # data.columns = data.loc['App']
    # data = data.drop('App', axis=0)
    data = data.set_index("App")
    print(data)
    data = figurePlotter.merge_columns(data, configs_groups, configs_groups_names)
    data = figurePlotter.exclude_and_sort_data(data, row_dict=apps_dict, column_dict=configs_dict)
    data = figurePlotter.rename_data(data, row_dict=apps_dict, column_dict=configs_dict)
    print("Processed data:\n", data)
    data = normalize_data(data, "BAP")
    print("Normalized data:\n", data)

    apps_labels = data.index.tolist()
    print("apps:", apps_labels)
    configs_labels = data.keys().values.tolist()
    print("configs_label:", configs_labels)

    # save_to_csv(data, figurePath)

    colorPalette = [
        "#ffdc6d",
        "#a0cc82",
        "#4c95cb",
        "#f19b61",
        "#ae8dca",
        "#c1c1c1",
        "#93bfcf",
        "#3fcfad",
    ]
    colorHatch = ["", "..", "x", "/", "\\", ":", "--", ","]
    figurePlotter.bar(
        apps_labels,
        configs_labels,
        data.values,
        plotSize=(15, 2.2),
        filename=figurePath,
        groupsInterval=0.15,
        colorPalette=colorPalette,
        colorHatch=colorHatch,
        xyConfig={
            "xylabel": ["", ylabel],
            "xlim": [None, None],
            "ylim": [0, 10],
            "labelExceedYlim": True,
            "xyscale": [None, None],
            "showxyTicksLabel": [True, True],
            "xyticksRotation": [30, 0],
            "xyticksMajorLocator": [None, 1],
        },
        averageConfig={
            "plotAverage": True,
            "onlyAverage": False,
            "labelAverage": True,
            "xlabel": "Gmean",
            "averageFunc": geo_mean,
            "labelExceedYlim": True,
        },
        legendConfig={
            "position": "lower center",
            "positionOffset": (0.45, 1),
            "col": 10,
            "legend.columnspacing": 1,
            "legend.handlelength": 2,
            "legend.handletextpad": 0.8,
        },
    )

if __name__ == "__main__":
    os.chdir(os.path.split(os.path.realpath(__file__))[0])
    # result_folder = "../ref_results/"
    result_folder = "../results/"
    path1 = result_folder+"raw/throughput_gpu_nap_breakdown/"
    paths = []
    paths.append(path1)
    # paths.append(path2)
    plot(path_list=paths,
         figurePath=result_folder+"fig14_breakdown.pdf",
         ylabel="Throughput\nNormalized to BAP")
