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

def normalize_data(data1, data2, normalize_to_column_name):
  row_names = data1.index.tolist()
  # error_value = 0
  for row_name in row_names:
    normalize_to = data1.loc[row_name, normalize_to_column_name] + data2.loc[row_name, normalize_to_column_name]
    if normalize_to <= 0:
      data1.loc[row_name] = np.nan
      data2.loc[row_name] = np.nan
    else:
      data1.loc[row_name] = data1.loc[row_name] / normalize_to
      data2.loc[row_name] = data2.loc[row_name] / normalize_to
    data1.loc[row_name][data1.loc[row_name] < 0] = np.nan
    data2.loc[row_name][data2.loc[row_name] < 0] = np.nan
    
  return data1, data2

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


def plot(path1, path2, figurePath, ylabel, ylim, normalize=False):
  # Load data
  data = pd.DataFrame()
  def load_data(path):
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
    # if data.empty:
    #   data = data_apps
    # else:
    #   data = data.merge(data_apps, how='outer', on = "App")
    return data_apps
  loads_data = load_data(path1)
  stores_data = load_data(path2)
  
  # print(data)
  # data.columns = data.loc['App']
  # data = data.drop('App', axis=0)  
  def proc_data(data):
    data = data.set_index('App')
    
    data = dp.merge_columns_min(data, configs_groups, configs_groups_names)
    data = dp.exclude_and_sort_data(
            data,  row_dict=apps_dict,  column_dict=configs_dict)
    data = dp.rename_data(data, row_dict=apps_dict,  column_dict=configs_dict)
    print("Processed data:\n", data)
    return data
  loads_data = proc_data(loads_data)
  stores_data = proc_data(stores_data)
  
  loads_data,  stores_data= normalize_data(loads_data, stores_data, "BAP")
  print("Normalized loads_data:\n", loads_data)
  print("Normalized stores_data:\n", stores_data)

  apps_labels = loads_data.index.tolist()
  print("apps:", apps_labels)
  # configs_labels = loads_data.keys().values.tolist()
  configs_labels = ['BAP', 'ngAP', 'ngAP+$\mathregular{O^1}$', 'ngAP+$\mathregular{O^2}$', 'ngAP+$\mathregular{O^3}$']
  stack_labels = ['Store', 'Load']
  
  print("configs_label:", configs_labels)

  # save_to_csv(data, figurePath)

  colorPalette = sns.color_palette("Blues", 1)
  colorPalette2 = sns.color_palette("YlOrBr", 2)
  colorPalette2 = ['#4c95cb']
  colorPalette = ['#a0cc82']
  colorHatch = ['', '//', 'xx', '..', '\\', '+', '--']
  mp.bar(apps_labels, configs_labels, stores_data.values, ylabel, filename=figurePath+"_avg.pdf", groupsInterval=0.15, labelExceedYlim=True,
         plotSize=(3.75, 2.2), 
         ylim=ylim, 
         yscale=None, colorPalette=colorPalette, legendCol = 5,
        #  colorHatch = colorHatch, 
         yMultipleLocator = 0.5,
         averageXlabel="GeoMean", averageFunc=geo_mean,
         stack=True, values2=loads_data.values, colorPalette2 = colorPalette2, stack_labels = stack_labels,
         only_average=True,
         decimals=2,
         ticksFrontsize=14, ticksRotation=30,
         plotHline = False)
  
  colorPalette = ['#ffdc6d', '#a0cc82', '#4c95cb', '#f19b61', '#ae8dca']
  mp.bar(apps_labels, configs_labels, stores_data.values, ylabel, filename=figurePath+".pdf", groupsInterval=0.15, labelExceedYlim=True,
        plotSize=(16, 3), 
        ylim=ylim, 
        yscale=None, colorPalette=colorPalette, legendCol = 5,
      #  colorHatch = colorHatch, 
        yMultipleLocator = 0.5,
        averageXlabel="GeoMean", averageFunc=geo_mean,
        stack=True, values2=loads_data.values, colorPalette2 = colorPalette2, stack_labels = stack_labels,
        # only_average=True,
        decimals=2,
        ticksFrontsize=14, ticksRotation=45)




if __name__ == "__main__":
    os.chdir(os.path.split(os.path.realpath(__file__))[0])

    path1 = "./results/raw/ncu/memory-loads"
    path2 = "./results/raw/ncu/memory-stores"
    # paths = []
    # paths.append(path1)
    # paths.append(path2)
    plot(path1, path2,
         figurePath="./results/ncu-memory-stack",
         ylabel="# of Memory Requests\nNormalized to BAP", 
         ylim=(0, 2),
         normalize = True)
    
    
    

