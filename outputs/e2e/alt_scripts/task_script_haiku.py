#!/usr/bin/env python3
"""
Analyze GOES-16 ABI-L2-CMIPC brightness temperature time series
for five named locations across bands 08, 09, and 10.
"""

import xarray as xr
import numpy as np
import pandas as pd
import glob
import os
from pathlib import Path
from datetime import datetime
import re

# Define locations (grid row, col)
LOCATIONS = {
    'Houston': (457, 690),
    'Atlanta': (241, 1306),
    'Dallas': (306, 663),
    'Nashville': (124, 1202),
    'Oklahoma City': (177, 672),
}

# Box size (10x10 pixels centered on location)
BOX_SIZE = 5  # ±5 pixels from center

def parse_filename(filename):
    """
    Parse GOES ABI filename to extract band and timestamp.
    Format: OR_ABI-L2-CMIPC-M6C{band}_G16_s{start_time}_e{end_time}_c{creation_time}.nc
    """
    basename = os.path.basename(filename)
    
    # Extract band
    band_match = re.search(r'C(\d{2})', basename)
    if not band_match:
        return None, None
    band = int(band_match.group(1))
    
    # Extract start time (format: YYYYDDDHHmmss.s)
    time_match = re.search(r's(\d{14})', basename)
    if not time_match:
        return None, None
    
    time_str = time_match.group(1)
    year = int(time_str[0:4])
    doy = int(time_str[4:7])
    hour = int(time_str[7:9])
    minute = int(time_str[9:11])
    second = int(time_str[11:13])
    
    # Convert day-of-year to datetime
    dt = datetime(year, 1, 1) + pd.Timedelta(days=doy-1, hours=hour, minutes=minute, seconds=second)
    
    return band, dt

def extract_brightness_temp(filename, location_name, row, col):
    """
    Extract brightness temperature from a 10x10 pixel box centered on (row, col).
    Keep only pixels where DQF == 0 (good quality).
    Return spatial mean brightness temperature.
    """
    try:
        ds = xr.open_dataset(filename)
        
        # Extract 10x10 box
        row_start = max(0, row - BOX_SIZE)
        row_end = min(ds.dims['y'], row + BOX_SIZE + 1)
        col_start = max(0, col - BOX_SIZE)
        col_end = min(ds.dims['x'], col + BOX_SIZE + 1)
        
        cmi = ds['CMI'].values[row_start:row_end, col_start:col_end]
        dqf = ds['DQF'].values[row_start:row_end, col_start:col_end]
        
        # Keep only good quality pixels (DQF == 0)
        good_mask = (dqf == 0)
        good_cmi = cmi[good_mask]
        
        if len(good_cmi) == 0:
            return None
        
        # Return spatial mean
        return float(np.mean(good_cmi))
        
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None

def main():
    # Find all NetCDF files
    data_dir = os.environ.get('E2E_DATA_DIR', '/data/goes_cmi_composites/raw/2024')
    files = sorted(glob.glob(f'{data_dir}/*/*/*/*.nc'))
    
    print(f"Found {len(files)} NetCDF files")
    
    # Process files
    results = []
    file_count = 0
    
    for i, filepath in enumerate(files):
        if i % 500 == 0:
            print(f"Processing file {i+1}/{len(files)}")
        
        band, dt = parse_filename(filepath)
        if band is None or dt is None:
            continue
        
        # Only process bands 08, 09, 10
        if band not in [8, 9, 10]:
            continue
        
        file_count += 1
        
        # Extract brightness temperature for each location
        for location_name, (row, col) in LOCATIONS.items():
            bt = extract_brightness_temp(filepath, location_name, row, col)
            if bt is not None:
                results.append({
                    'location': location_name,
                    'band': band,
                    'datetime_utc': dt,
                    'brightness_temp_K': bt,
                })
    
    print(f"Processed {file_count} files")
    print(f"Extracted {len(results)} brightness temperature measurements")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save to CSV
    output_csv = os.environ.get('E2E_OUTPUT_DIR', '/output/result') + '/goes_cmi_timeseries.csv'
    df.to_csv(output_csv, index=False)
    print(f"Saved CSV to {output_csv}")
    print(f"CSV shape: {df.shape}")
    print(f"CSV head:\n{df.head()}")
    
    return df, file_count

if __name__ == '__main__':
    df, file_count = main()
