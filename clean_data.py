import os
import pandas as pd
import numpy as np

def clean_folder_data(folder_path, task_keyword):
    """
    Scans a folder, aggregates all matching TSV files, handles shortforms
    by renaming them, drops useless duration columns, and reports exact instances.
    """
    total_raw_rows = 0
    total_missing_dropped = 0
    total_outliers_dropped = 0
    total_shortforms_renamed = 0
    
    combined_data = []
    
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' not found.")
        return pd.DataFrame()
        
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.tsv') and task_keyword in file_name:
            file_path = os.path.join(folder_path, file_name)
            
            # 1. Load file treating 'n/a' strings natively as numerical NaN
            df = pd.read_csv(file_path, sep='\t', na_values='n/a')
            total_raw_rows += len(df)
            
            # 2. Drop the useless duration column to save memory
            df = df.drop(columns=['duration'], errors='ignore')
            
            if task_keyword == 'PVT':
                # PVT specific adjustments
                df['response_time'] = df['response_time'] * 1000
                
                prev_len = len(df)
                df = df.dropna(subset=['response_time'])
                total_missing_dropped += (prev_len - len(df))
                
                # Filter false starts (<100ms)
                prev_len = len(df)
                df = df[df['response_time'] >= 100]
                total_outliers_dropped += (prev_len - len(df))
                
            elif task_keyword == 'SART':
                df['response_time'] = df['response_time'] * 1000
                
                # RETAIN & RENAME: Find 'rt' shortforms and rewrite them as 'response_action'
                rt_mask = df['value'] == 'rt'
                total_shortforms_renamed += rt_mask.sum()
                df.loc[rt_mask, 'value'] = 'response_action'
                
                # Filter false starts (<100ms)
                # (Note: we use a temporary mask so we don't drop NaNs belonging to misses!)
                invalid_rt = (df['response_time'] < 100) & (df['response_time'].notna())
                total_outliers_dropped += invalid_rt.sum()
                df = df[~invalid_rt]
                
            elif task_keyword == 'NB2':
                # NB2 response times are already in milliseconds
                
                # RETAIN & RENAME: Find 'rt' shortforms and rewrite them as 'response_action'
                rt_mask = df['value'] == 'rt'
                total_shortforms_renamed += rt_mask.sum()
                df.loc[rt_mask, 'value'] = 'response_action'
                
                # Filter false starts (<100ms)
                invalid_rt = (df['response_time'] < 100) & (df['response_time'].notna())
                total_outliers_dropped += invalid_rt.sum()
                df = df[~invalid_rt]
                
            combined_data.append(df)
            
    if not combined_data:
        print(f"No TSV files found for {task_keyword} in '{folder_path}'.")
        return pd.DataFrame()
        
    final_df = pd.concat(combined_data, ignore_index=True)
    
    print(f"--- Summary for {task_keyword} ---")
    print(f"Total raw rows processed: {total_raw_rows}")
    if total_shortforms_renamed > 0:
        print(f"Preserved and renamed 'rt' shortforms to 'response_action': {total_shortforms_renamed}")
    if total_missing_dropped > 0:
        print(f"Dropped due to missing values (PVT only): {total_missing_dropped}")
    print(f"Dropped due to false starts (<100ms): {total_outliers_dropped}")
    print(f"Final clean instances remaining for training: {len(final_df)}\n")
    
    return final_df

# Paths to folders
pvt_folder = "real_data"
model2_folder = "real_data_model2"

# Run cleaning on folders
pvt_cleaned = clean_folder_data(pvt_folder, 'PVT')
sart_cleaned = clean_folder_data(model2_folder, 'SART')
nback_cleaned = clean_folder_data(model2_folder, 'NB2')

# Saving as .npy files
def save_as_npy(df, folder, base_name):
    if df.empty:
        return
    # Convert dataframe to a raw numpy array grid (forces all items to strings)
    numpy_array = df.to_numpy()
    save_path = os.path.join(folder, f"{base_name}.npy")
    np.save(save_path, numpy_array)
    print(f"Successfully saved as NumPy binary: {save_path}")

save_as_npy(pvt_cleaned, pvt_folder, "cleaned_master_pvt")
save_as_npy(sart_cleaned, model2_folder, "cleaned_master_sart")
save_as_npy(nback_cleaned, model2_folder, "cleaned_master_nback")