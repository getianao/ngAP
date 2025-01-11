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
    "before-nfacg_-v100": ["NFA-CG-v100", -2],
    "before-nfacg_": ["NFA-CG", 2],
    "before-newtran-nt_": ["NT", -3],
    "before-newtran-ntmac_": ["NT-MaC", -4],
    "before-hotstarttt_": ["HotStartTT", -5],
    "before-hotstart-nt_-v100": ["GPU-NFA-v100", -10],
    "before-hotstart-nt_": ["GPU-NFA", 10],
    "before-hotstart-ntmac_": ["HotStart-Mac", -7],
    "before-hyperscan_": ["HyperScan", -8],
    "before-runahead-cc4_-v100": ["AsyncAP-v100", -9],
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
    "oa-nonblocking-default-256-best": ["ngAP-default-256", -63.3],
    "oa-nonblocking-default-best-v100": ["ngAP-default-v100", -63.5],
    "oa-nonblocking-default-best-e2-v100": ["ngAP-default-v100", -63.5],
    "oa-nonblocking-default-best-e2": ["ngAP-default", 63.5],
    "oa-nonblocking-all-best-uncomp": ["ngAP-Best-Uncomp", -63.6],
    "oa-nonblocking-all-best-v100": ["ngAP-best-v100", -63.7],
    "oa-nonblocking-all-best-e2-v100": ["ngAP-best-v100", -63.7],
    "oa-nonblocking-all-best-e2": ["ngAP-best", 64],
    # "oa-nonblocking-all-best": ["ngAP-best", 64],
    
    "o0-blocking-breakdown_": ["BAP", -81],
    "o0-nonblocking-NAP-breakdown_": ["ngAP", --82],
    "o1-nonblocking-breakdown_": ["ngAP+$\mathregular{O^1}$", -83],
    # "o4-nonblocking-r-breakdown_": ["NAP+O3", -84],
    "o3-nonblocking-p-breakdown_": ["ngAP+$\mathregular{O^2}$", -85],
    "oa-nonblocking-all-breakdown_": ["ngAP+$\mathregular{O^3}$", -86],
}


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

def normalize_v100_data(data):
  for column_name in data.keys():
    if not column_name.endswith("-v100"):
      print("#############", column_name, column_name+'-v100', data.loc[:, column_name+'-v100'])
      data.loc[:, column_name] /= data.loc[:, column_name+'-v100']
  return data

def remove_error(data, value):
  row_names = data.index.tolist()
  # error_value = 0
  for row_name in row_names:
    data.loc[row_name][data.loc[row_name].isna()] = value
    data.loc[row_name][data.loc[row_name] < 0] = value
  return data

def save_to_csv(data, csv_path):
  csv_file = os.path.splitext(os.path.abspath(csv_path))[0] + '.csv'
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
        csv_files = glob.glob(os.path.abspath(path)+'/*.{}'.format('csv'))
        for file in csv_files:
            df = pd.read_csv(file)
            if df.empty:
                continue
            df = df.T
            df.columns = df.loc["config"]
            df = df.drop('config', axis=0)
            if 'v100' in path:
                df = df.rename(columns=lambda x: x + '-v100')
            df["App"] = df.index.tolist()
            data_apps = pd.concat([data_apps, df])
            # print(data_apps, '\n')

        print(data_apps)

        if data.empty:
            data = data_apps
        else:
            data = data.merge(data_apps, how='outer', on = "App")

    # data.columns = data.loc['App']
    # data = data.drop('App', axis=0)
    data = data.set_index('App')
    print(data)
    
    data = normalize_v100_data(data)
    print("Normalized data:\n", data)
    print(data.columns)
    data = figurePlotter.exclude_and_sort_data(
          data,  row_dict=apps_dict_small,  column_dict=configs_dict)
    data = figurePlotter.rename_data(data, row_dict=apps_dict_small,  column_dict=configs_dict)
    data = remove_error(data, 0.16)
    print("Processed data:\n", data)
    # save_to_csv(data, figurePath)
    

    apps_labels = data.index.tolist()
    print("apps:", apps_labels)
    configs_labels = data.keys().values.tolist()
    print("configs_label:", configs_labels)

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
        plotSize=(5 * 2.5, 1* 2.5),
        filename=figurePath,
        groupsInterval=0.15,
        colorPalette=colorPalette,
        colorHatch=colorHatch,
        xyConfig={
            "xylabel": ["", ylabel],
            "xlim": [None, None],
            "ylim": [0, 4],
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


    path1 = "../results/raw_v100/throughput_gpu_nap_best_e2/"
    path2 = "../results/raw_v100/throughput_gpu_nap_default_adp_e2"
    path3 = "../results/raw_v100/throughput_gpu_sota_best/"
    path4 = "../results/raw_v100/throughput_gpu_runahead/"
    
    path5 = "../results/raw/throughput_gpu_nap_best_e2/"
    path6 = "../results/raw/throughput_gpu_nap_default_adp_e2"
    path7 = "../results/raw/throughput_gpu_sota_best/"
    path8 = "../results/raw/throughput_gpu_runahead/"
    
    paths = []
    paths.append(path1)
    paths.append(path2)
    paths.append(path3)
    paths.append(path4)
    paths.append(path5)
    paths.append(path6)
    paths.append(path7)
    paths.append(path8)

    plot(path_list=paths,
         figurePath="../results/throughput_gpu_ngap_v100_3090_o4.pdf",
         ylabel="Throughput\nNormalized to V100")
