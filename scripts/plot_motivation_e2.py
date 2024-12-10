import sys
import os
import pandas as pd
import numpy as np
import seaborn as sns
import glob
import figurePlotter
import figurePlotter.data_processing as dp

# from dict_config import apps_dict


configs_dict = {
    "self_loop_edge_perc": ["self edge", 1],
    "sequence_node_perc": ["sequential edge", 2],
    "all": ["other", 3],
}

apps_dict = {
    # AutomataZoo
    "APPRNG4": ["APR", 1],
    "Brill": ["Brill", 2],
    "CRISPR_CasOFFinder": ["CRP1", 3],
    "CRISPR_CasOT": ["CRP2", 4],
    "smallClamAV": ["CAV'", -5],  # 4degrees, 256states
    "ClamAV": ["CAV", 6],  # 4degrees,  # must behind small*
    "EntityResolution": ["ER", 7],
    "FileCarving": ["FC", -8],  # 4degrees, 256states
    "smallFileCarving": ["FC'", -9],  # 4degrees, 256states
    "Hamming_N1000_l18_d3": ["HM", 10],
    "Hamming_N1000_l22_d5": ["HM2", -11],
    "Hamming_N1000_l31_d10": ["HM3", -12],
    "Levenshtein_l19d3": ["LV", 13],
    "Levenshtein_l24d5": ["LV2", -14],
    "Protomata": ["Pro", 15],
    "RandomForest_20_400_200": ["RF", 16],
    "RandomForest_20_400_270": ["RF2", -17],
    "RandomForest_20_800_200": ["RF3", -18],
    "SeqMatch_BIBLE_w6_p6": ["SM", 19],
    "SeqMatch_BIBLE_w6_p10": ["SM2", -20],
    "smallSnort": ["Snort'", 21],  # 4degrees, 256states
    "Snort": ["Snort", 22],  # 4degrees, 256states
    "YARA": ["YARA", 23],  # 256states
    # ANMLZoo
    "Dotstar": ["DS", 31],
    "Fermi": ["Fermi", -32],
    "PowerEN": ["PEN", 33],
    # Regex
    "Bro217": ["Bro", 41],
    "ExactMath": ["EM", 42],
    "Ranges1": ["Ran1", 43],
    "Ranges05": ["Ran5", 44],
    "TCP": ["TCP", 45],
}


def set_datatype(data):
    cols = data.select_dtypes(exclude=["float"]).columns
    data[cols] = data[cols].apply(pd.to_numeric, downcast="float", errors="coerce")
    return data


def plot(path_list, figurePath, ylabel):
    data = pd.DataFrame()
    for path in path_list:
        print(path)
        data_apps = pd.DataFrame()
        csv_files = glob.glob(os.path.abspath(path) + "/*.{}".format("csv"))
        for file in csv_files:
            df = pd.read_csv(file)
            if df.empty:
                continue
            # df = df.T
            print(df)
            # df.columns = df.loc["name"]
            # df = df.drop("name", axis=0)
            df["App"] = df["name"]
            data_apps = pd.concat([data_apps, df])
            # print(data_apps, '\n')
        if data.empty:
            data = data_apps
        else:
            data = data.merge(data_apps, how="outer", on="App")

    data = data.set_index("App")
    data = set_datatype(data)
    data["all"] = 1
    print(data)

    data = dp.exclude_and_sort_data(
        data, row_dict=apps_dict, column_dict=configs_dict
    )
    data = dp.rename_data(
        data, row_dict=apps_dict, column_dict=configs_dict
    )
    data = data * 100
    print("Processed data:\n", data)
    # data = normalize_data(data, "Blocking")
    # print("Normalized data:\n", data)

    apps_labels = data.index.tolist()
    print("apps:", apps_labels)
    configs_labels = data.keys().values.tolist()
    print("configs_label:", configs_labels)

    # save_to_csv(data, figurePath)

    # colorPalette = ['#e6eef3', '#bbd9e8', '#7db6d4', '#408abb', '#2260a0']
    colorPalette = ["#2260a0", "#7db6d4", "#d6e4ec"]
    # colorPalette = ["#d6e4ec", "#7db6d4", "#2260a0"]
    colorPalette2 = sns.color_palette("YlOrBr", 3)
    # colorHatch = ['', '//', 'xx', '..', '\\', '+', '--']
    figurePlotter.stack(
        apps_labels,
        configs_labels,
        data.values,
        ylabel,
        filename=figurePath,
        groupsInterval=0.15,
        labelExceedYlim=True,
        plotSize=(7.5, 2.2),
        ylim=[0, 100],
        yscale=None,
        colorPalette=colorPalette,
        #  colorHatch = colorHatch,
        fontSize=12,
        yMultipleLocator=20,
        legendCol=5,
        averageXlabel="Gmean",
        averageFunc=dp.geo_mean,
        ticksRotation=45,
        decimals=1,
        legendPositionOffset=(0.5, 1),
        legendConfig={
            "legend.columnspacing": 0.9,
            "legend.handlelength": 2,
            "legend.handletextpad": 0.8,
        },
        # plotHline=False,
    )


if __name__ == "__main__":
    os.chdir(os.path.split(os.path.realpath(__file__))[0])
    result_folder = "../results/"
    path1 = result_folder + "raw/motivation_e2"
    paths = []
    paths.append(path1)
    plot(
        path_list=paths,
        figurePath=result_folder+"motivation_e2.pdf",
        ylabel="Edge Categories (%)",
    )
