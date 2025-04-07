import rasterio
import pandas as pd
from rasterio.transform import xy
import numpy as np
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Update this path to the location where you unzipped the PRISM bundle
bil_path = os.path.join(BASE_DIR, 'data/raw/temporal_data_showcase/PRISM_tdmean_stable_4kmM3_189501_bil.bil')

with rasterio.open(bil_path) as src:
    data = src.read(1)  # Read first band
    crs = src.crs
    transform = src.transform
    nodata = src.nodata

# Replace nodata with NaN
data = np.where(data == nodata, np.nan, data)

# Get row/col indices
rows, cols = np.meshgrid(np.arange(data.shape[0]), np.arange(data.shape[1]), indexing='ij')

# Convert row/col indices to lat/lon
lats, lons = xy(transform, rows, cols)

# Flatten everything into 1D arrays
df = pd.DataFrame({
    'latitude': np.array(lats).flatten(),
    'longitude': np.array(lons).flatten(),
    'value': data.flatten()
})

# Quick summary
print("Shape:", data.shape)
print("CRS:", crs)
print("Transform:", transform)

# Plot the temperature map
plt.imshow(data, cmap='coolwarm')
plt.colorbar(label='Temperature (°C)')
plt.title("PRISM Temperature - Jan 1895")
plt.show()

# Optional: Drop NaNs
df = df.dropna()

# Preview
print(df.head())