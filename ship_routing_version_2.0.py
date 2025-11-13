import os
import math
import random
import time as t
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import networkx as nx
import streamlit as st
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io import shapereader
from shapely.geometry import shape, Point, LineString
from shapely.prepared import prep
from shapely.strtree import STRtree
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from pathlib import Path

# ----------------
script_dir = Path(__file__).resolve().parent
shp_source_path_obj = script_dir / r"ne_50m_land\ne_50m_land.shp"
shp_source = str(shp_source_path_obj)


# Reduced graph parameters (faster)
GRID_LAT = 20
GRID_LON = 20
K_NEIGH = 4
INTERP_PER_EDGE = 8
MAX_ANIM_FRAMES = 300

# ----------------
class Ship:
    def __init__(self, name, speed, fuel_rate, wave_tolerance):
        self.name = name
        self.speed = speed
        self.fuel_rate = fuel_rate
        self.wave_tolerance = wave_tolerance

# ----------------
def haversine_nm(lat1, lon1, lat2, lon2):
    R_nm = 3440.065
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ = math.radians(lat2 - lat1)
    dλ = math.radians(lon2 - lon1)
    a = math.sin(dφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2
    return 2 * R_nm * math.asin(math.sqrt(a))

def great_circle_segment(lat1, lon1, lat2, lon2, n=8):
    φ1, λ1 = math.radians(lat1), math.radians(lon1)
    φ2, λ2 = math.radians(lat2), math.radians(lon2)
    Δσ = 2*math.asin(math.sqrt(
        math.sin((φ2-φ1)/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin((λ2-λ1)/2)**2
    ))
    if Δσ == 0:
        return [(lat1, lon1)]*n
    pts = []
    for k in range(n+1):
        t_ = k/float(n)
        A = math.sin((1-t_)*Δσ) / math.sin(Δσ)
        B = math.sin(t_*Δσ) / math.sin(Δσ)
        x = A*math.cos(φ1)*math.cos(λ1) + B*math.cos(φ2)*math.cos(λ2)
        y = A*math.cos(φ1)*math.sin(λ1) + B*math.cos(φ2)*math.sin(λ2)
        z = A*math.sin(φ1) + B*math.sin(φ2)
        φi = math.atan2(z, math.sqrt(x*x + y*y))
        λi = math.atan2(y, x)
        pts.append((math.degrees(φi), math.degrees(λi)))
    return pts

# ----------------
@st.cache_resource(show_spinner=False)
def load_land_geometries_resource(shp_path):
    # Accept folder, .shp, or zipped shapefile
    if os.path.isdir(shp_path):
        shps = glob.glob(os.path.join(shp_path, "*.shp"))
        if not shps:
            raise FileNotFoundError(f"No .shp files found in directory: {shp_path}")
        shp_path = shps[0]
    if not os.path.exists(shp_path):
        raise FileNotFoundError(f"Shapefile not found: {shp_path}")

    reader = shapereader.Reader(shp_path)
    geoms = [shape(g) for g in reader.geometries()]   # raw shapely geometries
    prepared = [prep(g) for g in geoms]               # prepared geometries
    strtree = STRtree(geoms) if geoms else None       # spatial index
    return prepared, strtree, reader, geoms

def point_is_land(lat, lon, prepared_geoms):
    p = Point(lon, lat)
    return any(pg.contains(p) for pg in prepared_geoms)

def segment_crosses_land_strtree(u, v, prepared_geoms, strtree, raw_geoms=None):
    seg = LineString([(u[1], u[0]), (v[1], v[0])])
    if strtree:
        candidates = strtree.query(seg)
        # candidates may be geometry objects or integer indices depending on Shapely version
        for c in candidates:
            if isinstance(c, (int, np.integer)):
                if raw_geoms is None:
                    continue
                geom = raw_geoms[int(c)]
            else:
                geom = c
            if getattr(geom, "intersects", None) and geom.intersects(seg):
                return True
        return False
    if prepared_geoms:
        return any(pg.intersects(seg) for pg in prepared_geoms)
    return False

# ----------------
def get_weather_value(lat, lon, weather_matrix, lat_range, lon_range):
    rows, cols = len(weather_matrix), len(weather_matrix[0])
    i = int((lat - lat_range[0]) / (lat_range[1] - lat_range[0]) * rows)
    j = int((lon - lon_range[0]) / (lon_range[1] - lon_range[0]) * cols)
    i = max(0, min(rows - 1, i))
    j = max(0, min(cols - 1, j))
    return weather_matrix[i][j]

# ----------------
@st.cache_resource(show_spinner=False)
def build_ocean_graph_reduced_resource(lat_range, lon_range, grid_steps_lat, grid_steps_lon, k_neighbors, shp_path):
    prepared_geoms, strtree, _, raw_geoms = load_land_geometries_resource(shp_path)
    lats = np.linspace(lat_range[0], lat_range[1], grid_steps_lat)
    lons = np.linspace(lon_range[0], lon_range[1], grid_steps_lon)
    all_nodes = [(float(lat), float(lon)) for lat in lats for lon in lons]
    ocean_nodes = [p for p in all_nodes if not point_is_land(p[0], p[1], prepared_geoms)]
    G = nx.Graph()
    for n in ocean_nodes:
        G.add_node(n)
    lon_step = (lon_range[1] - lon_range[0]) / float(grid_steps_lon)
    lat_step = (lat_range[1] - lat_range[0]) / float(grid_steps_lat)
    window_deg = max(lat_step * 6, lon_step * 6)
    for u in ocean_nodes:
        candidates = [v for v in ocean_nodes if abs(u[0]-v[0]) <= window_deg and abs(u[1]-v[1]) <= window_deg and v != u]
        if len(candidates) < k_neighbors:
            candidates = [v for v in ocean_nodes if v != u]
        dists = sorted(((u[0]-v[0])**2 + (u[1]-v[1])**2, v) for v in candidates)
        for _, v in dists[:k_neighbors]:
            if segment_crosses_land_strtree(u, v, prepared_geoms, strtree, raw_geoms):
                continue
            dist_nm = haversine_nm(u[0], u[1], v[0], v[1])
            G.add_edge(u, v, base_dist=dist_nm, weather_cost=0.0, weight=dist_nm)
    return G

def apply_weather_penalties(G, weather_matrix, ship, lat_range, lon_range):
    for (u, v) in G.edges():
        at_u = get_weather_value(u[0], u[1], weather_matrix, lat_range, lon_range)
        at_v = get_weather_value(v[0], v[1], weather_matrix, lat_range, lon_range)
        avg = (at_u + at_v) / 2.0
        penalty = (10 if avg > ship.wave_tolerance else 1) * ship.fuel_rate
        G.edges[u, v]['weather_cost'] = penalty
        G.edges[u, v]['weight'] = G.edges[u, v]['base_dist'] + penalty

def compute_stats(path, ship, G):
    total_nm = 0.0
    total_penalty = 0.0
    for u, v in zip(path[:-1], path[1:]):
        e = G.edges[u, v]
        total_nm += e.get('base_dist', 0.0)
        total_penalty += e.get('weather_cost', 0.0)
    time_hours = total_nm / ship.speed if ship.speed > 0 else float('inf')
    return total_nm, time_hours, total_penalty

def plot_image_marker(ax, lon, lat, image_data, zoom=0.075):
    im = OffsetImage(image_data, zoom=zoom)
    ab = AnnotationBbox(im, (lon, lat), frameon=False, pad=0.0, xycoords=ccrs.PlateCarree()._as_mpl_transform(ax))
    ax.add_artist(ab)

# ----------------
st.title("Ship routing")

lat_range = (20.0, 24.0)
lon_range = (88.0, 92.0)

ships = {
    "Cargo": Ship("Cargo", speed=15, fuel_rate=2, wave_tolerance=4),
    "Tanker": Ship("Tanker", speed=12, fuel_rate=3, wave_tolerance=5),
    "Passenger": Ship("Passenger", speed=20, fuel_rate=4, wave_tolerance=3),
}
ship_choice = st.sidebar.selectbox("Select Ship Type", list(ships.keys()))
ship = ships[ship_choice]
ship.speed = st.sidebar.number_input("Enter speed (knots)", min_value=1, max_value=30, value=ship.speed)
ship.fuel_rate = st.sidebar.number_input("Enter fuel rate", min_value=1, max_value=10, value=ship.fuel_rate)
ship.wave_tolerance = st.sidebar.slider("Wave tolerance (m)", 1, 10, ship.wave_tolerance)

weather_options = {
    "Calm Seas": [[1 for _ in range(30)] for _ in range(30)],
    "Corner Storm": [[random.randint(6,10) if i+j > 40 else 2 for j in range(30)] for i in range(30)],
    "Central Storm": [[random.randint(6,9) if 12 < i < 18 and 12 < j < 18 else random.randint(1,3) for j in range(30)] for i in range(30)],
    "Patchy": [[random.choice([1,2,3,7,8,9]) for _ in range(30)] for _ in range(30)],
    "Gradient": [[min(10, int(10*(i+j)/(58))) for j in range(30)] for i in range(30)],
}
scenario_choice = st.sidebar.selectbox("Select Weather Scenario", list(weather_options.keys()))
weather = weather_options[scenario_choice]

start_lat = st.sidebar.number_input("Start lat", value=21.2, format="%.4f")
start_lon = st.sidebar.number_input("Start lon", value=89.2, format="%.4f")
end_lat   = st.sidebar.number_input("End lat", value=22.8, format="%.4f")
end_lon   = st.sidebar.number_input("End lon", value=90.8, format="%.4f")
start = (start_lat, start_lon)
end   = (end_lat, end_lon)

# ----------------
try:
    prepared_geoms, strtree, shp_reader, raw_geoms = load_land_geometries_resource(shp_source)
except Exception as e:
    st.error(f"Failed loading shapefile {shp_source}: {e}")
    st.stop()

G = build_ocean_graph_reduced_resource(lat_range, lon_range, GRID_LAT, GRID_LON, K_NEIGH, shp_source)

def ensure_ocean_point(lat, lon, prepared_geoms, step=0.05, max_tries=20):
    tries = 0
    while point_is_land(lat, lon, prepared_geoms) and tries < max_tries:
        lon += step
        tries += 1
    return (lat, lon)

start = ensure_ocean_point(*start, prepared_geoms)
end   = ensure_ocean_point(*end, prepared_geoms)

for p in [start, end]:
    if p not in G.nodes():
        G.add_node(p)
    ocean_nodes = [n for n in G.nodes() if n != p]
    dists = [(haversine_nm(p[0], p[1], v[0], v[1]), v) for v in ocean_nodes]
    dists.sort(key=lambda x: x[0])
    for dist_nm, v in dists[:K_NEIGH]:
        if segment_crosses_land_strtree(p, v, prepared_geoms, strtree, raw_geoms):
            continue
        G.add_edge(p, v, base_dist=dist_nm, weather_cost=0.0, weight=dist_nm)

apply_weather_penalties(G, weather, ship, lat_range, lon_range)

try:
    path_nodes = nx.shortest_path(G, source=start, target=end, weight='weight')
except nx.NetworkXNoPath:
    st.error("No ocean-only path found. Try different start/end or relax constraints.")
    st.stop()

smooth_path = []
for u, v in zip(path_nodes[:-1], path_nodes[1:]):
    seg_pts = great_circle_segment(u[0], u[1], v[0], v[1], n=INTERP_PER_EDGE)
    if smooth_path:
        smooth_path.extend(seg_pts[1:])
    else:
        smooth_path.extend(seg_pts)

distance_nm, time_hours, fuel_units = compute_stats(path_nodes, ship, G)
st.write(f"**{ship.name} Ship Results**")
st.write(f"Route distance: {distance_nm:.1f} nautical miles")
st.write(f"Travel time: {time_hours:.2f} hours")
st.write(f"Fuel (penalty units): {fuel_units:.1f}")

# ----------------
# Static plot
fig, ax = plt.subplots(figsize=(8,8), subplot_kw={'projection': ccrs.PlateCarree()})
try:
    from cartopy.feature import ShapelyFeature
    land_feature = ShapelyFeature(list(shp_reader.geometries()), ccrs.PlateCarree(), facecolor='lightgray', edgecolor='black')
    ax.add_feature(land_feature)
except Exception:
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'))
    ax.add_feature(cfeature.BORDERS.with_scale('50m'))

ax.set_extent([lon_range[0], lon_range[1], lat_range[0], lat_range[1]])
lats = np.linspace(lat_range[0], lat_range[1], len(weather))
lons = np.linspace(lon_range[0], lon_range[1], len(weather[0]))
lon_grid, lat_grid = np.meshgrid(lons, lats)
hm = ax.pcolormesh(lon_grid, lat_grid, weather, cmap=plt.cm.Blues, vmin=0, vmax=10, transform=ccrs.PlateCarree())
plt.colorbar(hm, ax=ax, label="Wave height (m)")
if smooth_path:
    lats_path, lons_path = zip(*smooth_path)
    ax.plot(lons_path, lats_path, color='red', marker='o', transform=ccrs.PlateCarree())
st.pyplot(fig)

# ----------------
# Animation (limited frames)
script_dir = Path(__file__).resolve().parent
image_path_obj = script_dir / r"ship.jpg"
image_path = str(image_path_obj)

try:
    image_data = mpimg.imread(image_path)
except Exception:
    image_data = None

st.header("Animated simulation (limited frames)")
placeholder = st.empty()
max_frames = min(len(smooth_path), MAX_ANIM_FRAMES)

try:
    from cartopy.feature import ShapelyFeature
    land_feature_once = ShapelyFeature(list(shp_reader.geometries()), ccrs.PlateCarree(), facecolor='lightgray', edgecolor='black')
except Exception:
    land_feature_once = None

for i in range(1, max_frames+1):
    fig_anim, ax_anim = plt.subplots(figsize=(8, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    if land_feature_once:
        ax_anim.add_feature(land_feature_once)
    else:
        ax_anim.add_feature(cfeature.COASTLINE.with_scale('50m'))
        ax_anim.add_feature(cfeature.BORDERS.with_scale('50m'))
    ax_anim.set_extent([lon_range[0], lon_range[1], lat_range[0], lat_range[1]])
    ax_anim.pcolormesh(lon_grid, lat_grid, weather, cmap=plt.cm.Blues, vmin=0, vmax=10, transform=ccrs.PlateCarree())
    lats_path, lons_path = zip(*smooth_path[:i])
    ax_anim.plot(lons_path, lats_path, color='red', marker='o', markersize=3, transform=ccrs.PlateCarree())
    lat_cur, lon_cur = smooth_path[i-1]
    if image_data is not None:
        plot_image_marker(ax_anim, lon_cur, lat_cur, image_data, zoom=0.06)
    else:
        ax_anim.plot(lon_cur, lat_cur, marker='o', markersize=8, color='black', transform=ccrs.PlateCarree())
    placeholder.pyplot(fig_anim)
    plt.close(fig_anim)
    t.sleep(0.06)