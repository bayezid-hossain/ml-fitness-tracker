import pandas as pd
from glob import glob

# --------------------------------------------------------------
# Read single CSV file
# --------------------------------------------------------------

# single_acc_file = pd.read_csv(
#     "../../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv"
# )

# single_gyr_file = pd.read_csv(
#     "../../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv"
# )
# # --------------------------------------------------------------
# # List all data in data/raw/MetaMotion
# # --------------------------------------------------------------

# files = glob("../../data/raw/MetaMotion/*.csv")
# len(files)

# # --------------------------------------------------------------
# # Extract features from filename
# # --------------------------------------------------------------
# data_path = "../../data/raw/MetaMotion/"

# file = files[0].replace(data_path, "")
# contents = file.split("-")
# participant = contents[0]
# label = contents[1]
# category = contents[2].rstrip("123")

# df = pd.read_csv(files[0])

# df["participant"] = participant
# df["label"] = label
# df["category"] = category
# # --------------------------------------------------------------
# # Read all files
# # --------------------------------------------------------------

# acc_df = pd.DataFrame()
# gyr_df = pd.DataFrame()

# acc_set = 1
# gyr_set = 1

# for file in files:
#     contents = file.split("-")
#     participant = contents[0].replace(data_path, "")
#     label = contents[1]
#     category = contents[2].split("_")[0].rstrip("123")

#     df = pd.read_csv(file)

#     df["participant"] = participant
#     df["label"] = label
#     df["category"] = category

#     if "Accelerometer" in file:
#         df["set"] = acc_set
#         acc_set += 1
#         acc_df = pd.concat([acc_df, df])
#     else:
#         df["set"] = gyr_set
#         gyr_set += 1
#         gyr_df = pd.concat([gyr_df, df])
# --------------------------------------------------------------
# Working with datetimes
# --------------------------------------------------------------

# acc_df.info()
# pd.to_datetime(df["epoch (ms)"],unit="ms")
# acc_df.index=pd.to_datetime(acc_df["epoch (ms)"],unit="ms")
# gyr_df.index=pd.to_datetime(gyr_df["epoch (ms)"],unit="ms")

# del acc_df["epoch (ms)"]
# del gyr_df["epoch (ms)"]

# del acc_df["time (01:00)"]
# del gyr_df["time (01:00)"]

# del acc_df["elapsed (s)"]
# del gyr_df["elapsed (s)"]
# --------------------------------------------------------------
# Turn into function
# --------------------------------------------------------------


def read_files_from_csv(files):
    data_path = "../../data/raw/MetaMotion/"
    acc_df = pd.DataFrame()
    gyr_df = pd.DataFrame()

    acc_set = 1
    gyr_set = 1

    for file in files:
        contents = file.split("-")
        participant = contents[0].replace(data_path, "")
        label = contents[1]
        category = contents[2].split("_")[0].rstrip("123")

        df = pd.read_csv(file)

        df["participant"] = participant
        df["label"] = label
        df["category"] = category

        if "Accelerometer" in file:
            df["set"] = acc_set
            acc_set += 1
            acc_df = pd.concat([acc_df, df])
        else:
            df["set"] = gyr_set
            gyr_set += 1
            gyr_df = pd.concat([gyr_df, df])
    acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
    gyr_df.index = pd.to_datetime(gyr_df["epoch (ms)"], unit="ms")

    del acc_df["epoch (ms)"]
    del gyr_df["epoch (ms)"]

    del acc_df["time (01:00)"]
    del gyr_df["time (01:00)"]

    del acc_df["elapsed (s)"]
    del gyr_df["elapsed (s)"]

    return acc_df, gyr_df


files = glob("../../data/raw/MetaMotion/*.csv")
acc_df, gyr_df = read_files_from_csv(files)
# --------------------------------------------------------------
# Merging datasets
# --------------------------------------------------------------

data_merged = pd.concat([acc_df.iloc[:, :3], gyr_df], axis=1)

data_merged.columns = [
    "acc_x",
    "acc_y",
    "acc_z",
    "gyr_x",
    "gyr_y",
    "gyr_z",
    "participant",
    "label",
    "category",
    "set",
]
# --------------------------------------------------------------
# Resample data (frequency conversion)
# --------------------------------------------------------------

# Accelerometer:    12.500HZ
# Gyroscope:        25.000Hz

sampling = {
    "acc_x": "mean",
    "acc_y": "mean",
    "acc_z": "mean",
    "gyr_x": "mean",
    "gyr_y": "mean",
    "gyr_z": "mean",
    "participant": "last",
    "label": "last",
    "category": "last",
    "set": "last",
}

days = [g for n, g in data_merged.groupby(pd.Grouper(freq="D"))]

data_resampled = pd.concat(
    [df.resample(rule="200ms").apply(sampling).dropna() for df in days]
)
data_resampled["set"] = data_resampled["set"].astype("int")
data_resampled.info()
# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

data_resampled.to_pickle("../../data/interim/01_data_processed.pkl")
