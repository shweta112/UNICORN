import json
import random
from pathlib import Path
from typing import Iterable
from math import isnan
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from utils import replace_none_with_false
import os


def generate_dropout_list(x, p):
    n=len(x)
    values = [False] * n  # Initialize the list with n False values
    randint = random.randint(0, n - 1)
    values[randint] = True  # Set one random entry to True

    if x[randint] is None or x[randint].numel() == 0:
        return generate_dropout_list(x, p)
    
    for i in range(n):
        if not values[i]:  # For each remaining entry
            if random.random() < 1 - p:  # Generate a random number between 0 and 1
                values[i] = True  # Set the entry to True with probability p

    return values

def split_dataframe_by_patient(df: pd.DataFrame, n: int):
    """
    Splits a DataFrame into n random, non-overlapping subsets based on unique PATIENT_IDs.
    
    Parameters:
    - df (pd.DataFrame): The input DataFrame containing a 'PATIENT' column.
    - n (int): The number of splits required.
    
    Returns:
    - List[pd.DataFrame]: A list of n DataFrames, each containing a unique set of PATIENT_IDs.
    """
    # Extract PATIENT_ID from the PATIENT column
    df['PATIENT_ID'] = df['PATIENT'].apply(lambda x: x.split('_')[0])
    
    # Get unique PATIENT_IDs
    unique_patient_ids = df['PATIENT_ID'].unique()
    
    # Randomly shuffle the unique PATIENT_IDs
    np.random.shuffle(unique_patient_ids)
    
    # Split the shuffled PATIENT_IDs into n equal parts
    split_patient_ids = np.array_split(unique_patient_ids, n)
    
    # Initialize an empty list to store the n DataFrames
    df_splits = []
    
    # Create n DataFrames based on the split PATIENT_IDs
    for patient_ids in split_patient_ids:
        split_df = df[df['PATIENT_ID'].isin(patient_ids)]
        df_splits.append(split_df)
    
    # Remove the added PATIENT_ID column from the original DataFrame
    df.drop('PATIENT_ID', axis=1, inplace=True)
    
    return df_splits

def get_cohort_df(cfg) -> pd.DataFrame:
    cohort, cohort_data=cfg.cohort, cfg.cohort_data
    clini_table = cohort_data[cohort]["clini_table"]
    slide_csvs = cohort_data[cohort]["slide_csv"]
    feature_names=cohort_data[cohort]["cohort_features"]

    clini_df = (
        pd.read_csv(clini_table, dtype=str)
        if Path(clini_table).suffix == ".csv"
        else pd.read_excel(clini_table, dtype=str)
    )

    for i, (slide_csv,feature_name) in enumerate(zip(slide_csvs,feature_names)):
        feature_dir = Path(cohort_data[cohort]["feature_dir"][feature_name])
        slide_df = pd.read_csv(slide_csv, dtype=str)
        slide_df["slide_path"] = feature_dir / slide_df["FILENAME"]

        # Rename 'slide_path' and 'FILENAME' columns in slide_df
        slide_df = slide_df.rename(columns={"slide_path": f"slide_path_{i}", "FILENAME": f"FILENAME_{i}"})
        clini_df = (
            clini_df.merge(slide_df, on="PATIENT",how="left")
            .groupby("PATIENT")
            .first()
            .reset_index()
        )

    return clini_df


def group_files(folder_path):
    file_groups = {}

    for filename in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, filename)):
            parts = filename.split('_')
            if len(parts) >= 5:
                key = '_'.join(parts[0:1] + parts[3:5])  # Excluding "EvG" from the key
                file_groups.setdefault(key, []).append(filename)
            elif len(parts) == 3:
                key = parts[0] + "_" +parts[1]  # Excluding "EvG" from the key
                file_groups.setdefault(key, []).append(filename)

    return list(file_groups.values())

def custom_sort(filename_list, custom_order = ["vK", "Movat","HE", "EvG"]):
    def key_function(filename):
        for i, keyword in enumerate(custom_order):
            if keyword in filename:
                return i
        return len(custom_order)  # Default value if keyword not found

    sorted_filenames = sorted(filename_list, key=key_function)
    return sorted_filenames


class MILDatasetIndices(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        target_labels: Iterable[str],
        num_classes: int,
        clini_info: list = None,
        num_tiles: int = -1,
        pad_tiles: bool = True,
        norm: str = "macenko",
    ):
        self.data = data
        self.target_labels = target_labels
        self.clini_info = clini_info
        self.norm = norm
        self.num_tiles = num_tiles
        self.pad_tiles = pad_tiles
        self.num_classes = num_classes

    def __getitem__(self, item):
        # load features and coords from .h5 file

        patient_data = self.data.iloc[item]
        
        slide_path_keys = [
            slide_path
            for slide_path in patient_data.keys().values
            if "slide_path" in slide_path
        ]

        features = []
        coords = []
        extraction_args = []

        for slide_path_key in slide_path_keys:
            h5_path = patient_data[slide_path_key]
            #print(patient_data.PATIENT)
            #print(h5_path)
            if not (h5_path is None or (isinstance(h5_path, float) and isnan(h5_path))): 
                h5_file = h5py.File(h5_path)
                feature=torch.Tensor(np.array(h5_file["feats"]))
                coord=torch.Tensor(np.array(h5_file["coords"]))
                extraction_arg = h5_file["args"]
                extraction_arg = json.loads(extraction_arg[()].decode("UTF-8"))
                extraction_arg = replace_none_with_false(extraction_arg)
                extraction_arg["original_scene_sizes"] = np.array(h5_file["slide_sizes"])
            else:
                extraction_arg={}
                feature=torch.empty(0)
                coord=torch.empty(0)

            features.append(feature)
            coords.append(coord)
            extraction_args.append(extraction_arg)

        for additional_info in self.clini_info:
            if patient_data[additional_info] is not None:
                additional_feature= torch.ones((1,1)) * patient_data[additional_info]
            else: 
                additional_feature=torch.empty(0)
            features.append(additional_feature.to(torch.float32))

        # create numeric labels from categorical labels
        label = int(patient_data.TARGET)
        label = torch.eye(self.num_classes)[label]  # .squeeze(0)

        
        filename_keys = [key for key in self.data.keys() if "FILENAME" in key]
        filenames = [str(f) for f in patient_data[filename_keys].values] #replaces None by "None"

        return features, coords, label, patient_data.PATIENT, extraction_args, list(filenames)

    def __len__(self):
        return len(self.data)

def complete_file_list(files):
    # Define the expected endings
    endings = ['vK', 'Movat', 'HE', 'EvG']
    
    # Create an empty list to store the complete file names
    complete_files = []

    # Loop through the expected endings
    for ending in endings:
        # Find the file with the current ending, or set to None if not found
        file_with_ending = None
        for file in files:
            if ending in file:
                file_with_ending = file
                break
        
        # Append the found file or None to the complete_files list
        complete_files.append(file_with_ending)

    return complete_files

class InferenceDataset(Dataset):
    def __init__(
        self,
        folder: str
    ):
        self.data=[custom_sort(a) for a in group_files(folder)]
        self.data=[a for a in self.data if len(a)==4]
        self.folder=folder


    def __getitem__(self, item):

        features = []
        coords = []
        extraction_args = []
        slide_paths=self.data[item]
        if len(slide_paths)!=4:
            slide_paths=complete_file_list(slide_paths)

        for i,slide_path in enumerate(slide_paths):
            if slide_path is None:
                features.append(torch.Tensor([]))
                coords.append(torch.Tensor([]))
                extraction_args.append([])
                slide_paths[i]="None"
            else:
                h5_file = h5py.File(Path(self.folder)/slide_path)
                features.append(torch.Tensor(np.array(h5_file["feats"])))
                coords.append(torch.Tensor(np.array(h5_file["coords"])))

                extraction_arg = h5_file["args"]
                extraction_arg = json.loads(extraction_arg[()].decode("UTF-8"))
                extraction_arg = replace_none_with_false(extraction_arg)

                extraction_arg["original_scene_sizes"] = np.array(h5_file["slide_sizes"])
                extraction_args.append(extraction_arg)
                patient=slide_path.split("_")[0]+"_"+slide_path.split("_")[1]

        return features, coords, torch.Tensor([0,0,0,0,1]), patient ,extraction_args,list(slide_paths)

    def __len__(self):
        return len(self.data)




def transform_clini_info(
    df: pd.DataFrame, cfg, desired_mean: np.ndarray, desired_std: np.ndarray
) -> pd.DataFrame:
    """transform columns with categorical features to integers and normalize them with given mean and std dev"""
    # fill missing columns with 0
    columns = df.columns.tolist()
    for info in columns:
        if info in cfg.clini_info:
            # only choose rows with valid labels
            valid_rows = df[info].astype(str).str.replace('.', '', regex=False).str.isdigit()
            col = df.loc[valid_rows, info]
            
            # map columns to integers
            if info == "SEGMENT":
                col = col.astype(int)
                label_list = []
                for entry in col:
                    label = np.zeros(7)
                    label[entry] = 7 * desired_mean
                    label_list.append(label)
                df.loc[valid_rows, info] = pd.Series(label_list)
                
            else:
                col = col.astype(float)
                # normalize columns
                mean = col.mean()
                std = col.std()
                col = (col - mean) / std
                col = col * desired_std + desired_mean  # Adjust mean and std to desired values
                
                # add normalized columns back to dataframe
                df.loc[valid_rows, info] = col
            # fill missing values with 0
            #df[label] = df[label].fillna(0)

