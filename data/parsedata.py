import pickle
import numpy as np
import pandas as pd

## Parse pkl data file
# open the pkl file
with open("./data/LMP_data.pkl", "rb") as f:
    # load the data from the pkl file
    data = pickle.load(f)

# do something with the data
selected_nodes = [
    "PGF1_2_PDRP88-APND",
    "HIDESER_1_N018",
    "SDG1_1_PDRP66-APND",
    "EDMON5AP_7_N002",
    "POD_COLTON_6_AGUAM1-APND",
    "EDMON1AP_7_N001",
    "MGM_LNODE-4",
    "RIO_LNODE21A",
    "EDMON7AP_7_N001",
    "WINTERWD_LNODE-2",
    "POD_SLRMS3_2_SRMSR1-APND",
    "POD_SKERN_6_SOLAR2-APND",
    "EDENVALE_1_N004",
    "VICTORVL_5_N101",
    "HG_LNODER2A",
    "POD_CNTNLA_2_SOLAR2-APND",
    "POD_SNMALF_6_UNITS-APND",
    "HICKS_2_N001",
    "EELRIVR_6_N001",
    "SDG1_1_PDRP72-APND",
    "PGNP_2_PDRP45-APND",
    "SDG1_1_PDRP86-APND",
    "VESTAL_6_N020",
    "PGNP_2_PDRP27-APND",
    "EAGLEMTN_2_B1",
    "SDG1_1_PDRP110-APND",
    "POD_CO_5_UNIT1-APND",
    "HIGHLINE_LNODELD3",
    "POD_CONTRL_1_POOLE-APND",
    "SDG1_1_PDRP41-APND",
    "ELMA_PGE_LNODEBR1"
]

df = pd.DataFrame()
for node in selected_nodes:
    temp = data[node]
    ls = []
    for i in range(365):
        ls+=list(temp[i])
    df[node] = ls

df.to_csv("./data/LMP_data.csv", index=False)