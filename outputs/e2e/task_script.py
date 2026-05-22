#!/usr/bin/env python3
"""
Process GOES-16 ABI-L2-CMIPC data to extract brightness temperature time series
at specified locations for bands 08, 09, and 10.
"""

import os
import glob
import re
from datetime import datetime
import numpy as np
import pandas as pd
import netCDF4 as nc
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Define locations (row, col)
LOCATIONS = {
    'Houston': (457, 690),
    'Atlanta': (241, 1306),
    'Dallas': (306, 663),
    'Nashville': (124, 1202),
    'Oklahoma City': (177, 672)
}

# Box size (10x10 centered on location)
BOX_HALF_SIZE = 5

def parse_goes_filename(filename):
    """
    Parse GOES ABI filename to extract band and timestamp.
    Format: OR_ABI-L2-CMIPC-M6C{band}_G16_s{start}_e{end}_c{created}.nc
    """
    basename = os.path.basename(filename)
    
    # Extract band number
    band_match = re.search(r'C(\d+)', basename)
    if not band_match:
        return None, None
    band = int(band_match.group(1))
    
    # Extract start time (s{year}{doy}{hour}{minute}{second}{subsec})
    time_match = re.search(r's(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})', basename)
    if not time_match:
        return None, None
    
    year = int(time_match.group(1))
    doy = int(time_match.group(2))
    hour = int(time_match.group(3))
    minute = int(time_match.group(4))
    second = int(time_match.group(5))
    
    # Convert day of year to datetime
    dt = datetime.strptime(f'{year}-{doy}', '%Y-%j')
    dt = dt.replace(hour=hour, minute=minute, second=second)
    
    return band, dt

def extract_box_mean(cmi_data, dqf_data, row, col, half_size=5):
    """
    Extract mean brightness temperature from a box centered on (row, col).
    Only include pixels where DQF == 0 (good quality).
    """
    # Define box boundaries
    row_start = max(0, row - half_size)
    row_end = min(cmi_data.shape[0], row + half_size)
    col_start = max(0, col - half_size)
    col_end = min(cmi_data.shape[1], col + half_size)
    
    # Extract box
    cmi_box = cmi_data[row_start:row_end, col_start:col_end]
    dqf_box = dqf_data[row_start:row_end, col_start:col_end]
    
    # Filter by quality flag
    good_pixels = cmi_box[dqf_box == 0]
    
    if len(good_pixels) == 0:
        return np.nan
    
    return np.mean(good_pixels)

def process_all_files(data_dir, output_csv):
    """
    Process all GOES files and extract time series for all locations.
    """
    # Find all NetCDF files for bands 08, 09, 10
    pattern = os.path.join(data_dir, '**', '*C0[8-9]*.nc')
    files_8_9 = glob.glob(pattern, recursive=True)
    
    pattern = os.path.join(data_dir, '**', '*C10*.nc')
    files_10 = glob.glob(pattern, recursive=True)
    
    all_files = files_8_9 + files_10
    all_files = sorted(all_files)
    
    print(f"Found {len(all_files)} files to process")
    
    results = []
    
    for i, filepath in enumerate(all_files):
        if i % 100 == 0:
            print(f"Processing file {i+1}/{len(all_files)}")
        
        # Parse filename
        band, dt = parse_goes_filename(filepath)
        if band is None or dt is None:
            print(f"Warning: Could not parse filename {filepath}")
            continue
        
        # Only process bands 8, 9, 10
        if band not in [8, 9, 10]:
            continue
        
        try:
            # Open NetCDF file
            with nc.Dataset(filepath, 'r') as ds:
                cmi = ds.variables['CMI'][:]
                dqf = ds.variables['DQF'][:]
                
                # Process each location
                for location_name, (row, col) in LOCATIONS.items():
                    mean_bt = extract_box_mean(cmi, dqf, row, col, BOX_HALF_SIZE)
                    
                    results.append({
                        'location': location_name,
                        'band': band,
                        'datetime_utc': dt,
                        'brightness_temp_K': mean_bt
                    })
        
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            continue
    
    # Create DataFrame and save
    df = pd.DataFrame(results)
    df = df.sort_values(['location', 'band', 'datetime_utc'])
    df.to_csv(output_csv, index=False)
    
    print(f"\nSaved {len(df)} records to {output_csv}")
    print(f"Unique locations: {df['location'].nunique()}")
    print(f"Unique bands: {sorted(df['band'].unique())}")
    print(f"Date range: {df['datetime_utc'].min()} to {df['datetime_utc'].max()}")
    print(f"Files processed: {len(all_files)}")
    
    return df

def create_diurnal_cycle_plot(df, output_png):
    """
    Create multi-panel plot showing diurnal cycle for each band.
    """
    # Extract hour of day
    df['hour'] = df['datetime_utc'].dt.hour + df['datetime_utc'].dt.minute / 60.0
    
    # Calculate hourly statistics
    hourly_stats = df.groupby(['location', 'band', 'hour'])['brightness_temp_K'].agg(['mean', 'std']).reset_index()
    
    # Create figure with 3 panels (one per band)
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle('GOES-16 ABI Brightness Temperature Diurnal Cycles\n(May 1-7, 2024 UTC)', 
                 fontsize=14, fontweight='bold')
    
    bands = [8, 9, 10]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for idx, band in enumerate(bands):
        ax = axes[idx]
        band_data = hourly_stats[hourly_stats['band'] == band]
        
        for loc_idx, location in enumerate(LOCATIONS.keys()):
            loc_data = band_data[band_data['location'] == location].sort_values('hour')
            
            if len(loc_data) > 0:
                ax.plot(loc_data['hour'], loc_data['mean'], 
                       label=location, color=colors[loc_idx], linewidth=2, marker='o', markersize=3)
                
                # Add shaded error region
                ax.fill_between(loc_data['hour'], 
                               loc_data['mean'] - loc_data['std'],
                               loc_data['mean'] + loc_data['std'],
                               alpha=0.2, color=colors[loc_idx])
        
        ax.set_xlabel('Hour of Day (UTC)', fontsize=11)
        ax.set_ylabel('Brightness Temperature (K)', fontsize=11)
        ax.set_title(f'Band {band:02d}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 3))
        
        if idx == 0:
            ax.legend(loc='best', fontsize=9, ncol=2)
    
    plt.tight_layout()
    plt.savefig(output_png, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to {output_png}")
    plt.close()

def create_report(df, output_md, num_files):
    """
    Create a markdown report documenting the analysis.
    """
    report = f"""# GOES-16 ABI Brightness Temperature Analysis Report

## Dataset Overview
- **Satellite**: GOES-16
- **Product**: ABI-L2-CMIPC (Cloud and Moisture Imagery - CONUS)
- **Period**: 2024-05-01 through 2024-05-07 UTC (7 days)
- **Bands**: 08, 09, 10 (infrared channels)
- **Temporal Resolution**: ~5-minute cadence
- **Total Files Processed**: {num_files}

## Locations Analyzed

The following five locations were analyzed using fixed grid coordinates:

| Location       | Grid Row | Grid Column |
|----------------|----------|-------------|
| Houston        | 457      | 690         |
| Atlanta        | 241      | 1306        |
| Dallas         | 306      | 663         |
| Nashville      | 124      | 1202        |
| Oklahoma City  | 177      | 672         |

## Methodology

### Spatial Averaging
For each location and timestamp, brightness temperature was extracted from a **10×10 pixel box** 
centered on the specified grid coordinates (±5 pixels in each direction).

### Quality Filtering
Only pixels with **DQF (Data Quality Flag) = 0** (good quality) were included in the spatial mean.
Pixels with poor quality flags were excluded from the calculation.

### Band Information
- **Band 08**: 6.2 μm (upper-level water vapor)
- **Band 09**: 6.9 μm (mid-level water vapor)
- **Band 10**: 7.3 μm (lower-level water vapor)

## Results Summary

### Data Records
- **Total records extracted**: {len(df):,}
- **Records per location**: {len(df) // 5:,} (average)
- **Date range**: {df['datetime_utc'].min().strftime('%Y-%m-%d %H:%M:%S')} to {df['datetime_utc'].max().strftime('%Y-%m-%d %H:%M:%S')}

### Brightness Temperature Statistics

"""
    
    # Add statistics by band and location
    for band in sorted(df['band'].unique()):
        band_data = df[df['band'] == band]
        report += f"\n#### Band {band:02d}\n\n"
        report += "| Location       | Mean (K) | Std Dev (K) | Min (K) | Max (K) | Valid Records |\n"
        report += "|----------------|----------|-------------|---------|---------|---------------|\n"
        
        for location in LOCATIONS.keys():
            loc_data = band_data[band_data['location'] == location]['brightness_temp_K'].dropna()
            if len(loc_data) > 0:
                report += f"| {location:14s} | {loc_data.mean():8.2f} | {loc_data.std():11.2f} | {loc_data.min():7.2f} | {loc_data.max():7.2f} | {len(loc_data):13,d} |\n"
    
    report += f"""

## Output Files

1. **goes_cmi_timeseries.csv**: Complete time series data with columns:
   - `location`: Location name
   - `band`: Band number (8, 9, or 10)
   - `datetime_utc`: Observation timestamp (UTC)
   - `brightness_temp_K`: Spatially-averaged brightness temperature in Kelvin

2. **goes_cmi_point_timeseries.png**: Multi-panel visualization showing diurnal cycles
   (hourly mean ± standard deviation) for all five locations across the 7-day period.

3. **report.md**: This report

## Notes

- Missing values (NaN) in the output indicate that no good-quality pixels were available 
  in the 10×10 box for that location and timestamp.
- The diurnal cycle plots show the mean and standard deviation computed across all days 
  in the analysis period for each hour of the day.
- All timestamps are in UTC.

---
*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC*
"""
    
    with open(output_md, 'w') as f:
        f.write(report)
    
    print(f"Saved report to {output_md}")

def main():
    """Main processing function."""
    data_dir = os.environ.get('E2E_DATA_DIR', '/data/goes_cmi_composites/raw/2024')
    output_dir = '' + os.environ.get('E2E_OUTPUT_DIR', '/output/result') + ''
    
    output_csv = os.path.join(output_dir, 'goes_cmi_timeseries.csv')
    output_png = os.path.join(output_dir, 'goes_cmi_point_timeseries.png')
    output_md = os.path.join(output_dir, 'report.md')
    
    # Process all files
    print("=" * 80)
    print("GOES-16 ABI Brightness Temperature Extraction")
    print("=" * 80)
    df = process_all_files(data_dir, output_csv)
    
    # Count actual files processed
    num_files = len(glob.glob(os.path.join(data_dir, '**', '*C0[8-9]*.nc'), recursive=True))
    num_files += len(glob.glob(os.path.join(data_dir, '**', '*C10*.nc'), recursive=True))
    
    # Create visualizations
    print("\n" + "=" * 80)
    print("Creating Diurnal Cycle Plot")
    print("=" * 80)
    create_diurnal_cycle_plot(df, output_png)
    
    # Create report
    print("\n" + "=" * 80)
    print("Generating Report")
    print("=" * 80)
    create_report(df, output_md, num_files)
    
    print("\n" + "=" * 80)
    print("Processing Complete!")
    print("=" * 80)
    print(f"Output files:")
    print(f"  - {output_csv}")
    print(f"  - {output_png}")
    print(f"  - {output_md}")

if __name__ == '__main__':
    main()
