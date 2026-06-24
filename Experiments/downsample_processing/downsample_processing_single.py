
import pandas as pd


def read_csv_data(csv_data_path):
    return pd.read_csv(csv_data_path)



def main():
    adt_dataset_path = "/mnt/d/SparseGaze/ADT-Gaze-structured/sequences"
    sequence_name = "Apartment_release_decoration_skeleton_seq131_M1292"
    # Read Gaze data : gaze/gaze_samples.csv
    gaze_data_path = f"{adt_dataset_path}/{sequence_name}/gaze/gaze_samples.csv"
    head_data_path = f"{adt_dataset_path}/{sequence_name}/head/head_samples.csv"
    gaze_data = read_csv_data(gaze_data_path)
    head_data = read_csv_data(head_data_path)
    original_fps = 30
    target_fps = 5 # 15, 10, 6, 5
    downsample_factor = original_fps // target_fps
    warmup_frames = 15 # Number of initial frames to skip for stable FPS estimation

    # 现在gaze数据有哪些features了？我们先看看gaze_samples.csv的前几行
    print("Original gaze data columns:", gaze_data.columns)
    print("Original head data columns:", head_data.columns)
    print("Downsampling gaze data...")
    downsampled_gaze_data = gaze_data.iloc[warmup_frames::downsample_factor].reset_index(drop=True)
    downsampled_head_data = head_data.iloc[warmup_frames::downsample_factor].reset_index(drop=True)



if __name__ == "__main__":
    main()