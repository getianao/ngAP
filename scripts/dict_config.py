apps_dict = {
    # AutomataZoo
    "APPRNG4": ["APR", 1],
    "Brill": ["Brill", 2],
    "CRISPR_CasOFFinder": ["CRP1", 3],
    "CRISPR_CasOT": ["CRP2", 4],
    "smallClamAV": ["CAV\'", 5],          # 4degrees, 256states
    "ClamAV": ["CAV", -6],          # 4degrees,  # must behind small*
    "EntityResolution": ["ER", 7],
    "FileCarving": ["FC", -8],     # 4degrees, 256states
    "smallFileCarving": ["FC\'", -9],     # 4degrees, 256states
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
    "smallSnort": ["Snort\'", 21],       # 4degrees, 256states
    "Snort": ["Snort", -22],       # 4degrees, 256states
    "YARA": ["YARA", 23],         # 256states
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

apps_dict_small = {
    # AutomataZoo
    "APPRNG4": ["APR", 1],
    "Brill": ["Brill", 2],
    "CRISPR_CasOFFinder": ["CRP1", 3],
    "CRISPR_CasOT": ["CRP2", 4],
    "smallClamAV": ["CAV\'", 5],          # 4degrees, 256states
    "ClamAV": ["CAV", -6],          # 4degrees,  # must behind small*
    "EntityResolution": ["ER", 7],
    "FileCarving": ["FC", -8],     # 4degrees, 256states
    "smallFileCarving": ["FC\'", -9],     # 4degrees, 256states
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
    "smallSnort": ["Snort\'", 21],       # 4degrees, 256states
    "Snort": ["Snort", -22],       # 4degrees, 256states
    "YARA": ["YARA", 23],         # 256states
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

apps_dict2 = {
    # AutomataZoo
    "APPRNG4": ["APR", 1],
    "Brill": ["Brill", 2],
    "CRISPR_CasOFFinder": ["CRP1", 3],
    "CRISPR_CasOT": ["CRP2", 4],
    "smallClamAV": ["CAV\'", -5],          # 4degrees, 256states
    "ClamAV": ["CAV", 6],          # 4degrees,  # must behind small*
    "EntityResolution": ["ER", 7],
    "FileCarving": ["FC", -8],     # 4degrees, 256states
    "smallFileCarving": ["FC\'", -9],     # 4degrees, 256states
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
    "smallSnort": ["Snort\'", -21],       # 4degrees, 256states
    "Snort": ["Snort", 22],       # 4degrees, 256states
    "YARA": ["YARA", 23],         # 256states
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

apps_dict_small2 = {
    # AutomataZoo
    "APPRNG4": ["APR", 1],
    "Brill": ["Brill", 2],
    "CRISPR_CasOFFinder": ["CRP1", 3],
    "CRISPR_CasOT": ["CRP2", 4],
    "smallClamAV": ["CAV\'", -5],          # 4degrees, 256states
    "ClamAV": ["CAV", 6],          # 4degrees,  # must behind small*
    "EntityResolution": ["ER", 7],
    "FileCarving": ["FC", -8],     # 4degrees, 256states
    "smallFileCarving": ["FC\'", -9],     # 4degrees, 256states
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
    "smallSnort": ["Snort\'", -21],       # 4degrees, 256states
    "Snort": ["Snort", 22],       # 4degrees, 256states
    "YARA": ["YARA", 23],         # 256states
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

parameters_dict = {
    "oa-nonblocking-all-aasinterval-32_": ["32", -101],
    "oa-nonblocking-all-aasinterval-64_": ["64", -102],
    "oa-nonblocking-all-aasinterval-128_": ["128", -103],
    "oa-nonblocking-all-aasinterval-256_": ["256", 104],
    "oa-nonblocking-all-aasinterval-512_": ["512", 105],
    "oa-nonblocking-all-aasinterval-1024_": ["1024", 106],
    "oa-nonblocking-all-aasinterval-5120_": ["5120", -107],
    "oa-nonblocking-all-aasinterval-10240_": ["10240", -108],
    "oa-nonblocking-all-aasinterval-adp_": ["adaptive", 109],
    


    # "oa-nonblocking-all-continuous-1r32_": ["0%", 111],
    # "oa-nonblocking-all-continuous-1r24_": ["25%", 112],
    # "oa-nonblocking-all-continuous-1r16_": ["50%", 113],
    # "oa-nonblocking-all-continuous-1r8_": ["75%", -114],
    # "oa-nonblocking-all-continuous-1r0_": ["100%", 115],
    
    "oa-nonblocking-all-continuous-1r0_": ["0%", 115],
    "oa-nonblocking-all-continuous-1r8_": ["25%", 114],
    "oa-nonblocking-all-continuous-1r16_": ["50%", 113],
    "oa-nonblocking-all-continuous-1r24_": ["75%", -112],
    "oa-nonblocking-all-continuous-1r32_": ["100%", 111],
    
    
    
    
    
    "oa-nonblocking-all-fetchsize-256_": ["256", 121],
    # "oa-nonblocking-all-fetchsize-1024_": ["1024", 122],
    "oa-nonblocking-all-fetchsize-5120_": ["5120", 123],
    "oa-nonblocking-all-fetchsize-10240_": ["10240", 124],
    "oa-nonblocking-all-fetchsize-25600_": ["25600", 124.5],
    "oa-nonblocking-all-fetchsize-40960_": ["40960", -125],
    "oa-nonblocking-all-fetchsize-102400_": ["102400", -126],
    
    "oa-nonblocking-all-precompute-0_": ["0", 131],
    "oa-nonblocking-all-precompute-1_": ["1", 132],
    "oa-nonblocking-all-precompute-2_": ["2", 133],
    "oa-nonblocking-all-precompute-3_": ["3", 134],
}

