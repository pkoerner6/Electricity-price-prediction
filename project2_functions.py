import numpy as np
import pandas as pd
from typing import Dict, List
import math



def get_mean_based_on_season(df: pd.DataFrame): 
    df_mean = df.groupby(["season"]).mean()
    return df_mean.to_dict(orient="index")

def replace_nans_with_mean(df: pd.DataFrame):
    season_to_dict = get_mean_based_on_season(df)
    for i in df.index:
        season = df.at[i, "season"]
        for col in df.columns:
            if col == "season":
              continue
            if math.isnan(df.at[i, col]):
                season_dict = season_to_dict[season]
                mean = season_dict[col]
                df.at[i, col] = mean
    return df

def encode_categorical_data(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical data using principles of the OrdinalEncoder()"""
    cat_dict_season = {"spring": 0, "summer": 1, "autumn": 2, "winter": 3}
    df = df.replace(
        {
            "season": cat_dict_season,
        }
    )
    return df

