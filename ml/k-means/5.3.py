import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from shapely.geometry import Point
from scipy.interpolate import griddata
from datetime import datetime, timedelta

# Load the GeoJSON file
gdf = gpd.read_file(r"D:\desk\vs\web-system-skeleton\frontend\public\data\final_heatwaves.geojson")
# Define the time range (modify as needed)
start_date = datetime(2020, 6, 1)
end_date = datetime(2020, 8, 31)

# Filter events within the time range
gdf['start_date'] = pd.to_datetime(gdf['start_date'])
gdf = gdf[(gdf['start_date'] >= start_date) & (gdf['start_date'] <= end_date)]

# Function to extract centroid and cumulative anomaly
def extract_centroid_anomaly(row):
    geom = row['geometry']
    centroid = geom.centroid  # Get the centroid of the event's geometry
    return pd.Series({
        'lon': centroid.x,
        'lat': centroid.y,
        'cumulative_anomaly': row['cumulative_anomaly']
    })

# Extract centroids and cumulative anomalies
centroid_data = gdf.apply(extract_centroid_anomaly, axis=1)

# Create a grid for interpolation
lon_min, lon_max = centroid_data['lon'].min(), centroid_data['lon'].max()
lat_min, lat_max = centroid_data['lat'].min(), centroid_data['lat'].max()
lon_grid = np.linspace(lon_min, lon_max, 100)
lat_grid = np.linspace(lat_min, lat_max, 100)
lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

# Prepare data for interpolation
points = centroid_data[['lon', 'lat']].values
values = centroid_data['cumulative_anomaly'].values

# Interpolate cumulative anomaly onto the grid
z_grid = griddata(points, values, (lon_mesh, lat_mesh), method='cubic')

# Replace NaN values with 0 for visualization
z_grid = np.nan_to_num(z_grid, nan=0.0)

# Create the 3D surface plot
fig = go.Figure(data=[
    go.Surface(
        x=lon_grid,
        y=lat_grid,
        z=z_grid,
        colorscale='Viridis',
        colorbar=dict(title='Cumulative Anomaly (°C)'),
        showscale=True
    )
])

# Update layout
fig.update_layout(
    title=f"3D Surface Plot of Marine Heatwave Cumulative Intensity ({start_date.date()} to {end_date.date()})",
    scene=dict(
        xaxis_title='Longitude (°E)',
        yaxis_title='Latitude (°N)',
        zaxis_title='Cumulative Anomaly (°C)',
        xaxis=dict(range=[lon_min, lon_max]),
        yaxis=dict(range=[lat_min, lat_max]),
        zaxis=dict(range=[0, z_grid.max() * 1.1]),  # Adjust z-axis range
        aspectratio=dict(x=1, y=1, z=0.5)
    ),
    width=800,
    height=600
)

# Show the plot
fig.show()

# Save the plot as HTML
fig.write_html("mhws_3d_surface.html")