import sys
import os
import pandas as pd
import numpy as np
import glob
import seaborn as sns
import figure_plotting.myplot as mp
import figure_plotting.data_processing as dp
from dict_config import *

configs_dict = {
    # sota
    "before-infant_": ["iNFAnt", -1],
    "before-nfacg_": ["NFA-CG", 2],
    "before-newtran-nt_": ["NT", -3],
    "before-newtran-ntmac_": ["NT-MaC", -4],
    "before-hotstarttt_": ["HotStartTT", -5],
    "before-hotstart-nt_": ["HotStart", 10],
    "before-hotstart-ntmac_": ["HotStart-Mac", -7],
    "before-hyperscan_": ["HyperScan", -8],
    "before-runahead-cc4_": ["Runahead", 9],
    
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
    "oa-nonblocking-all-best": ["NAP-Best", -64],
    

    
    "o0-blocking-breakdown_": ["BAP", 81],
    "o0-nonblocking-NAP-breakdown_": ["ngAP", 82],
    "o1-nonblocking-breakdown_": ["ngAP+$\mathregular{O^1}$", 83],
    # "o4-nonblocking-r-breakdown_": ["NAP+O3", -84],
    "o3-nonblocking-p-breakdown_": ["ngAP+$\mathregular{O^2}$", 85],
    "oa-nonblocking-all-breakdown_": ["ngAP+$\mathregular{O^3}$", 86],
}

configs_groups = [["o0-blocking_"], ["o0-nonblocking-NAP_"], ["o1-nonblocking_"], ["o4-nonblocking-r1_", "o4-nonblocking-r1f_", "o4-nonblocking-r2_", "o4-nonblocking-r2f_"],
                  ["o3-nonblocking-p1_", "o3-nonblocking-p2_", "o3-nonblocking-p3_"], ["oa-nonblocking-all-p2r1_", "oa-nonblocking-all-p2r1f_", "oa-nonblocking-all-p3r1_", "oa-nonblocking-all-p3r1f_"]]
configs_groups_names = ["o0-blocking-breakdown_", "o0-nonblocking-NAP-breakdown_", "o1-nonblocking-breakdown_",
                        "o4-nonblocking-r-breakdown_", "o3-nonblocking-p-breakdown_", "oa-nonblocking-all-breakdown_"]

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
  csv_file = os.path.splitext(os.path.abspath(csv_path))[0] + '.csv'
  print("Save data to", csv_file)
  data.to_csv(csv_file)
  

def geo_mean(x):
    a = np.log(x)
    return np.exp(a.mean())


def plot(path_list, figurePath, ylabel, ylim, normalize=False):
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
      df["App"] = df.index.tolist()
      data_apps = pd.concat([data_apps, df])
      # print(data_apps, '\n')
    if data.empty:
      data = data_apps
    else:
      data = data.merge(data_apps, how='outer', on = "App")
  
  print(data)
  # data.columns = data.loc['App']
  # data = data.drop('App', axis=0)  
  data = data.set_index('App')
  
  data = dp.merge_columns(data, configs_groups, configs_groups_names)
  data = dp.exclude_and_sort_data(
          data,  row_dict=apps_dict,  column_dict=configs_dict)
  data = dp.rename_data(data, row_dict=apps_dict,  column_dict=configs_dict)
  print("Processed data:\n", data)
  if normalize:
    data = normalize_data(data, "BAP")
    print("Normalized data:\n", data)

  apps_labels = data.index.tolist()
  print("apps:", apps_labels)
  # configs_labels = data.keys().values.tolist()
  configs_labels = ['BAP', 'ngAP', 'ngAP+$\mathregular{O^1}$', 'ngAP+$\mathregular{O^2}$', 'ngAP+$\mathregular{O^3}$']
  
  print("configs_label:", configs_labels)

  # save_to_csv(data, figurePath)

  colorPalette = ['#4c95cb']
  colorPalette2 = ['#a0cc82']
  colorHatch = ['', '//', 'xx', '..', '\\', '+', '--']
  mp.bar(apps_labels, configs_labels, data.values, ylabel, filename=figurePath+"_avg.pdf", groupsInterval=0.15, labelExceedYlim=True,
         plotSize=(3.75, 2.2), ylim=ylim, yscale=None, colorPalette=colorPalette,colorHatch = colorHatch, 
        #  yMultipleLocator =50,
         only_average=True,
         decimals = 2,
         averageXlabel="GeoMean", averageFunc=geo_mean,
         ticksFrontsize = 14, ticksRotation = 30,
         plotHline = False
         )
  
  colorPalette = ['#ffdc6d', '#a0cc82', '#4c95cb', '#f19b61', '#ae8dca']
  mp.bar(apps_labels, configs_labels, data.values, ylabel, filename=figurePath+".pdf", groupsInterval=0.15, labelExceedYlim=True,
         plotSize=(16, 3), ylim=ylim, yscale=None, colorPalette=colorPalette,colorHatch = colorHatch, 
        #  yMultipleLocator =50,
        #  only_average=True,
         decimals = 2,
         averageXlabel="GeoMean", averageFunc=geo_mean,
         ticksFrontsize = 14, ticksRotation = 30)




if __name__ == "__main__":
    os.chdir(os.path.split(os.path.realpath(__file__))[0])

    # path1 = "./results/raw/ncu/memory"
    # paths = []
    # paths.append(path1)
    # # paths.append(path2)
    # plot(path_list=paths,
    #      figurePath="./results/ncu-memory.pdf",
    #      ylabel="Global memory trasaction\n(Normalized to Blocking)", ylim=(0,50), normalize = True)
    
    
    path1 = "./results/raw/ncu/l1cache"
    paths = []
    paths.append(path1)
    # paths.append(path2)
    plot(path_list=paths,
         figurePath="./results/ncu-c1cache",
         ylabel="L1$ Hit Rate", ylim=(0,1))
