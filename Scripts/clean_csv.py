import os
import time
import pandas as pd
import sys

from utils import print_with_timestamp

def clean_csv(file_path):
    """
    Cleans and filters the contents of a CSV file.
    """
    print_with_timestamp("Script started")
    print_with_timestamp(f"Cleaning CSV file: {file_path}")

    start_time = time.time()

    # Read CSV file
    df = pd.read_csv(file_path)

    # Filter rows based on condition
    df = df[df['id.orig_h'].isin(['192.168.1.136'])]
    df = df[df['id.resp_h'].isin(['192.168.1.5'])]
    #df = df[~df['service'].isin(['mqtt'])]
    df = df[df['proto'] != 'proto']  # Exclude rows where 'proto' column has the value 'proto'

    # Drop unnecessary columns
    df = df.drop(['ts', 'uid', 'uid.1'], axis=1)

    # Write the cleaned contents back to the CSV file
    df.to_csv(file_path, index=False)

    end_time = time.time()
    execution_time = end_time - start_time
    print_with_timestamp(f"Script completed in {execution_time:.2f} seconds")

if len(sys.argv) != 2:
    print("Usage: python3 clean_csv.py <folder_path>")
    sys.exit(1)

folder_path = sys.argv[1]
clean_csv(folder_path)