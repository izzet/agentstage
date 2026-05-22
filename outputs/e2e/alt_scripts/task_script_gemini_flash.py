
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Define locations and their grid coordinates
locations = {
    "Houston": (457, 690),
    "Atlanta": (241, 1306),
    "Dallas": (306, 663),
    "Nashville": (124, 1202),
    "Oklahoma City": (177, 672),
}

# Define the bands to process
bands = [8, 9, 10]

# Output directory
output_dir = "/output/result"
os.makedirs(output_dir, exist_ok=True)

all_data = []
processed_files_count = 0

# Loop through days and hours
for day in range(122, 129):  # Days 122 to 128 (May 1st to May 7th)
    for hour in range(24):
        path = f"/data/goes_cmi_composites/raw/2024/{day:03d}/{hour:02d}/"
        if not os.path.exists(path):
            continue

        for filename in os.listdir(path):
            if filename.endswith(".nc"):
                filepath = os.path.join(path, filename)
                
                # Parse band and timestamp from filename
                parts = filename.split('_')
                band_str = parts[3][7:9] # e.g., M6C08 -> 08
                band = int(band_str)
                
                if band not in bands:
                    continue

                start_time_str = parts[4][1:] # sYYYYDDDHHMMSS0
                datetime_utc = pd.to_datetime(start_time_str, format='%Y%j%H%M%S')

                try:
                    with xr.open_dataset(filepath, engine="netcdf4") as ds:
                        for loc_name, (row, col) in locations.items():
                            # Define a 10x10 pixel box centered on the location
                            row_start = row - 5
                            row_end = row + 5
                            col_start = col - 5
                            col_end = col + 5

                            # Read CMI and DQF for the box
                            cmi_box = ds['CMI'].isel(y=slice(row_start, row_end), x=slice(col_start, col_end)).values
                            dqf_box = ds['DQF'].isel(y=slice(row_start, row_end), x=slice(col_start, col_end)).values

                            # Keep only pixels where DQF == 0 (good quality)
                            good_quality_cmi = cmi_box[dqf_box == 0]

                            # Compute spatial mean brightness temperature
                            if good_quality_cmi.size > 0:
                                mean_brightness_temp = np.nanmean(good_quality_cmi)
                            else:
                                mean_brightness_temp = np.nan # No good quality pixels

                            all_data.append({
                                "location": loc_name,
                                "band": band,
                                "datetime_utc": datetime_utc,
                                "brightness_temp_K": mean_brightness_temp,
                            })
                    processed_files_count += 1
                except Exception as e:
                    print(f"Error processing file {filepath}: {e}")

# Create DataFrame
df = pd.DataFrame(all_data)

# Save to CSV
output_csv_path = os.path.join(output_dir, "goes_cmi_timeseries.csv")
df.to_csv(output_csv_path, index=False)
print(f"CSV saved to {output_csv_path}")

# Generate summary figure
fig, axes = plt.subplots(nrows=len(bands), ncols=1, figsize=(12, 15), sharex=True)
fig.suptitle("Diurnal Cycle of Brightness Temperature (Hourly Mean ± Std)", fontsize=16)

for i, band in enumerate(bands):
    ax = axes[i]
    band_df = df[df['band'] == band].copy()
    band_df['hour'] = band_df['datetime_utc'].dt.hour

    hourly_stats = band_df.groupby(['location', 'hour'])['brightness_temp_K'].agg(['mean', 'std']).reset_index()

    for loc_name in locations.keys():
        loc_hourly_stats = hourly_stats[hourly_stats['location'] == loc_name]
        ax.plot(loc_hourly_stats['hour'], loc_hourly_stats['mean'], label=loc_name)
        ax.fill_between(loc_hourly_stats['hour'],
                        loc_hourly_stats['mean'] - loc_hourly_stats['std'],
                        loc_hourly_stats['mean'] + loc_hourly_stats['std'],
                        alpha=0.2)

    ax.set_title(f"Band {band}")
    ax.set_ylabel("Brightness Temperature (K)")
    ax.legend(loc='upper right')
    ax.grid(True)

axes[-1].set_xlabel("Hour of Day (UTC)")
plt.xticks(range(24))
plt.tight_layout(rect=[0, 0.03, 1, 0.96])
output_png_path = os.path.join(output_dir, "goes_cmi_point_timeseries.png")
plt.savefig(output_png_path)
print(f"Figure saved to {output_png_path}")

# Generate report
report_content = f"""
# GOES-16 ABI-L2-CMIPC Analysis Report

## Overview
This report summarizes the analysis of GOES-16 ABI-L2-CMIPC bundle data for 2024-05-01 through 2024-05-07 UTC. Brightness temperature time series were extracted at five named locations for bands 08, 09, and 10.

## Locations and Grid Coordinates
The following locations were analyzed, with their respective grid row and column coordinates:
- Houston: (row 457, col 690)
- Atlanta: (row 241, col 1306)
- Dallas: (row 306, col 663)
- Nashville: (row 124, col 1202)
- Oklahoma City: (row 177, col 672)

## Bands Processed
Data from the following ABI bands were processed:
- Band 08
- Band 09
- Band 10

## Quality Filtering
For each location and file, a 10x10 pixel box centered on the location was extracted for both CMI (brightness temperature) and DQF (data quality flag) variables. Only pixels with a Data Quality Flag (DQF) equal to 0 (indicating good quality data) were used in the calculation of the spatial mean brightness temperature for each box. If no good quality pixels were found, the mean brightness temperature was recorded as NaN.

## Files Processed
A total of {processed_files_count} NetCDF files were successfully processed.

## Outputs
1. **CSV File**: `goes_cmi_timeseries.csv`
   Contains the extracted brightness temperature time series with columns: `location`, `band`, `datetime_utc`, `brightness_temp_K`.
2. **Summary Figure**: `goes_cmi_point_timeseries.png`
   A multi-panel plot showing the diurnal cycle curves (hourly mean ± standard deviation over the 7-day period) for all five locations, with one panel per band.
"""

output_report_path = os.path.join(output_dir, "report.md")
with open(output_report_path, "w") as f:
    f.write(report_content)
print(f"Report saved to {output_report_path}")
