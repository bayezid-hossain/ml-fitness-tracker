import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle("../../data/interim/02_outliers_removed_chauvenet.pkl")

predictor_columns = list(df.columns[:6])
plt.style.use("fivethirtyeight")

plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2


# --------------------------------------------------------------
# Dealing with missing values (imputation)
# --------------------------------------------------------------

for col in predictor_columns:
    df[col] = df[col].interpolate()

df.info()
# --------------------------------------------------------------
# Calculating set duration
# --------------------------------------------------------------

duration = df[df["set"] == 1].index[-1] - df[df["set"] == 1].index[0]
duration.seconds

for s in df["set"].unique():
    start = df[df["set"] == s].index[-1]
    end = df[df["set"] == s].index[0]
    duration = start - end
    df.loc[(df["set"] == s), "duration"] = duration.seconds

duration_df = df.groupby(["category"])["duration"].mean()

duration_df.iloc[0] / 5
duration_df.iloc[1] / 10
# --------------------------------------------------------------
# Butterworth lowpass filter
# --------------------------------------------------------------

df_lowpass = df.copy()
lowpassFilter = LowPassFilter()

frequency = 1000 / 200
cutoff = 1.3

for col in predictor_columns:
    df_lowpass = lowpassFilter.low_pass_filter(
        col=col,
        cutoff_frequency=cutoff,
        data_table=df_lowpass,
        sampling_frequency=frequency,
    )
    df_lowpass[col] = df_lowpass[col + "_lowpass"]
    del df_lowpass[col + "_lowpass"]


# --------------------------------------------------------------
# Principal component analysis PCA
# --------------------------------------------------------------
df_pca = df_lowpass.copy()

pca = PrincipalComponentAnalysis()

pca_component_values = pca.determine_pc_explained_variance(
    cols=predictor_columns, data_table=df_pca
)
plt.figure(figsize=(10, 10))
plt.plot(range(1, len(predictor_columns) + 1), pca_component_values)
plt.xlabel("Principal Component count")
plt.ylabel("Explained variance")
plt.show()

df_pca = pca.apply_pca(cols=predictor_columns, data_table=df_pca, number_comp=3)

subset = df_pca[df_pca["set"] == 35]
subset[["pca_1", "pca_2", "pca_3"]].plot()

df_pca

# --------------------------------------------------------------
# Sum of squares attributes
# --------------------------------------------------------------

df_squared = df_pca.copy()

acc_r = df_squared["acc_x"] ** 2 + df_squared["acc_y"] ** 2 + df_squared["acc_z"] ** 2

gyr_r = df_squared["gyr_x"] ** 2 + df_squared["gyr_y"] ** 2 + df_squared["gyr_z"] ** 2

df_squared["acc_r"] = np.sqrt(acc_r)
df_squared["gyr_r"] = np.sqrt(gyr_r)

# subset = df_squared[(df_squared["category"] == "medium") & (df_squared["label"] == "squat") & (df_squared["participant"] == "D")]
subset = df_squared[df_squared["set"] == 48]
subset[["acc_r", "gyr_r"]].plot(subplots=True)
df_squared
# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------

df_temporal = df_squared.copy()

NumAbs = NumericalAbstraction()

predictor_columns = list(df.columns[:6]) + ["acc_r", "gyr_r"]
ws = int(1000 / 200)
for col in predictor_columns:
    df_temporal = NumAbs.abstract_numerical(
        data_table=df_temporal, aggregation_function="mean", cols=[col], window_size=ws
    )
    df_temporal = NumAbs.abstract_numerical(
        data_table=df_temporal, aggregation_function="std", cols=[col], window_size=ws
    )
list(df_temporal.columns)

df_temporal_list = []

for s in df_temporal["set"].unique():
    subset = df_temporal[df_temporal["set"] == s].copy()
    for col in predictor_columns:
        subset = NumAbs.abstract_numerical(
            data_table=subset, aggregation_function="mean", cols=[col], window_size=ws
        )
        subset = NumAbs.abstract_numerical(
            data_table=subset, aggregation_function="std", cols=[col], window_size=ws
        )
    df_temporal_list.append(subset)
df_temporal = pd.concat(df_temporal_list)
df_temporal.info()
subset[["acc_y", "acc_y_temp_mean_ws_5", "acc_y_temp_std_ws_5"]].plot()
subset[["gyr_y", "gyr_y_temp_mean_ws_5", "gyr_y_temp_std_ws_5"]].plot()
# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------
df_freq = df_temporal.copy().reset_index()
FreqAbs = FourierTransformation()

fs = int(1000 / 200)
ws = int(2800 / 200)

df_freq = FreqAbs.abstract_frequency(df_freq, ["acc_y"], ws, fs)


subset = df_freq[df_freq["set"] == 14]
subset[["acc_y"]].plot()
subset[
    [
        "acc_y_max_freq",
        "acc_y_freq_weighted",
        "acc_y_pse",
        "acc_y_freq_1.429_Hz_ws_14",
        "acc_y_freq_2.5_Hz_ws_14",
    ]
].plot()


# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------


# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------


# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------
