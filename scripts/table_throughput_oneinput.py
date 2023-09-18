import sys
import os
import pandas as pd
import numpy as np
import glob
import os
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
    "before-hyperscan_": ["HyperScan", 1.5],
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
    # "oa-nonblocking-default-32-best": ["NAP-default-32", 63.1],
    # "oa-nonblocking-default-128-best": ["NAP-default-128", 63.2],
    "oa-nonblocking-default-256-best": ["NAP-default-256", 63.3],
    "oa-nonblocking-default-best": ["NAP", 63.5],
    "oa-nonblocking-all-best": ["NAP-Best", 64],
    
    "o0-blocking-breakdown_": ["Blocking", 81],
    "o0-nonblocking-NAP-breakdown_": ["NAP", 82],
    "o1-nonblocking-breakdown_": ["NAP+O1", 83],
    # "o4-nonblocking-r-breakdown_": ["NAP+O3", -84],
    "o3-nonblocking-p-breakdown_": ["NAP+O2", 85],
    "oa-nonblocking-all-breakdown_": ["NAP+O3", 86],
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

def remove_nan(data, value):
  row_names = data.index.tolist()
  # error_value = 0
  for row_name in row_names:
    data.loc[row_name][data.loc[row_name].isna()] = value
  return data

def set_datatype(data):
  cols = data.select_dtypes(exclude=['float']).columns
  data[cols] = data[cols].apply(pd.to_numeric, downcast='float', errors='coerce')
  return data



def geo_mean(x):
    a = np.log(x)
    return np.exp(a.mean())



def merge_csv(path_list, save_path):
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
      df["App"] = df.index.tolist()
      data_apps = pd.concat([data_apps, df])
      # print(data_apps, '\n')
    if data_apps.empty:
      continue
    if data.empty:
      data = data_apps
    else:
      data = data.merge(data_apps, how='outer', on = "App")
  
  print(data)

  # data.columns = data.loc['App']
  # data = data.drop('App', axis=0)  
  data = data.set_index('App')
  data = figurePlotter.exclude_and_sort_data(
          data,  row_dict=apps_dict2,  column_dict=configs_dict)
  data = figurePlotter.rename_data(data, row_dict=apps_dict2,  column_dict=configs_dict)
  # data = data.T

  
  # data = normalize_data(data, "HotStart")
  # print("Normalized data:\n", data)
  # data = remove_nan(data, 0.01)
  # print("Final data:\n", data)
  
  
  data = set_datatype(data)
  # data = data.round(2)
  data = data.replace(np.nan, -3)

  print("Processed data:\n", data)
  data.to_csv(save_path, sep=',', index = True, float_format = '%.2f')
  
  df2 = pd.read_csv(save_path)
  df2 = df2.replace(-3, "U")
  df2 = df2.replace(-2, "T")
  df2 = df2.replace(-1, "W")
  # print(df2)
  df2.to_csv(save_path, sep=',', index = False)


if __name__ == "__main__":
    os.chdir(os.path.split(os.path.realpath(__file__))[0])
    
    # result_folder = "../ref_results/"
    result_folder = "../results/"
    path1 = result_folder+"raw/throughput_gpu_nap_best_oneinput"
    path2 = result_folder+"raw/throughput_gpu_sota_best_oneinput"
    path3 = result_folder+"raw/throughput_gpu_runahead_oneinput"
    path4 = result_folder+"raw/throughput_cpu_oneinput"
    path5 = result_folder+"raw/throughput_gpu_nap_default_adp_oneinput"
    paths = []
    paths.append(path1)
    paths.append(path2)
    paths.append(path3)
    paths.append(path4)
    paths.append(path5)
    save_path = result_folder+"tab6_latency.csv"
    
    merge_csv(paths, save_path)


