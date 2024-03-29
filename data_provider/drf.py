import numpy as np
import random
import pandas as pd

def reset_id(data):
    """
    Resets the car IDs in the given DataFrame to be consecutive integers.

    Args:
        data: A pandas DataFrame containing information about the cars, including their ID and frame.

    Returns:
        The input DataFrame with the car IDs reset to be consecutive integers.
    """
    car_id = 1
    # Sort the DataFrame by car ID and then by frame number
    data.sort_values(by=["car_id", "frame"], ascending=True)
    # Reset the index so that it starts at 0
    data = data.reset_index(drop=True)
    # Loop through each row in the DataFrame
    for i in range(len(data)):
        if i == 0:
            last_id = data["car_id"][i]
            data["car_id"][i] = car_id
        # If the current frame number is more than 1 greater than the previous frame number
        # or the current row has a different car ID than the previous row, increment the car ID
        elif data["frame"][i] - data["frame"][i-1] > 1 or data["car_id"][i] != last_id:
            car_id += 1
            last_id = data["car_id"][i]
            data["car_id"][i] = car_id
        # Otherwise, set the car ID to the current car ID
        else:
            data["car_id"][i] = car_id
    return data


def rand_del(data, del_num, del_length, min_interval):
    """
    Deletes a specified number of rows from a pandas DataFrame at random with a specified interval.

    Args:
        data: A pandas DataFrame to delete rows from.
        del_num: The number of rows to delete.
        interval: The interval at which to sample rows.

    Returns:
        The input DataFrame with the specified number of rows deleted at random with the specified interval.
    """
    # Get the number of rows in the DataFrame
    num_rows = len(data)
    # Generate a list of indices to sample from with the specified interval
    sample_indices = range(0, num_rows-del_length, min_interval)
    # Generate a list of random indices to delete from the sample indices
    print("del_num", del_num)
    del_indices = random.sample(sample_indices, del_num)
    # Delete the rows with the random indices
    for i in del_indices:
        seq = range(i, i + del_length)
        data = data.drop(index=seq)
    return data

def drf(gt_flnm, outflnm, drf_rate, del_length=40, min_interval=66):
    data = pd.read_csv(gt_flnm)
    data["gt_id"] = data["car_id"]
    del_num = int(len(data) * (1 - drf_rate) / del_length)
    data = rand_del(data, del_num, del_length, min_interval)
    data = reset_id(data)
    data["frame"] = data["frame"].astype('int')
    data["car_id"] = data["car_id"].astype('int')
    data["gt_id"] = data["gt_id"].astype('int')
    data.to_csv(outflnm, index=False)
    data.to_csv("data/img/RML7/drf%.2f-ori.csv"%drf_rate, index=False)


if __name__ == '__main__':

    for drf_rate in np.arange(0.4, 0.6, 0.05):
        gt_flnm = "data/img/RML7/packed.csv"
        outflnm = "data/img/RML7/drf%.2f.csv"%drf_rate
        data = drf(gt_flnm, outflnm, drf_rate)

