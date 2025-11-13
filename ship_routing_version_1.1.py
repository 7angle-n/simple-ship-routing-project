import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
import random
import streamlit as st
import time as t
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from pathlib import Path
import os

n_size = 3000
class Ship:
    def __init__(self, name, speed, fuel_rate, wave_tolerance):
        self.name = name
        self.speed = speed
        self.fuel_rate = fuel_rate
        self.wave_tolerance = wave_tolerance

def build_ocean_grid(width=10, height=10):
    G = nx.grid_2d_graph(width, height)
    for (u, v) in G.edges():
        G.edges[u, v]['weight'] = 1
    return G 

def compute_stats(path, ship):
    distance = len(path) - 1
    time = distance/ship.speed
    fuel = distance*ship.fuel_rate
    return distance, time, fuel

def shortest_path(G, start, end):
    path = nx.shortest_path(G, source=start, target=end, weight='weight')
    return path

def weather_changes_penalty(G, weather, ship):
    for (u, v) in G.edges():
        at_u = weather[u[0]][u[1]]
        at_v = weather[v[0]][v[1]]
        avg = (at_u + at_v)/2
        if avg > ship.wave_tolerance:
            G.edges[u, v]['weight'] = 10*ship.fuel_rate
        else:
            G.edges[u, v]['weight'] = 1*ship.fuel_rate

def plot_route_with_weather(G, path, ship, scenario):
    font1 = {'family':'serif','color':'blue','size':20}
    pos = {(x,y): (x,y) for x,y in G.nodes()}
    node_colors = [weather[x][y] for (x,y) in G.nodes()]
    fig, ax = plt.subplots(figsize=(8,8))
    nx.draw(G, pos, node_size=n_size, vmin=0, vmax=10, node_color=node_colors, with_labels=False, cmap=plt.cm.Blues, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=path, node_color="red", ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=list(zip(path[:-1], path[1:])), edge_color="red", width=2, ax=ax)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(vmin=0, vmax=10))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Wave height (m)")
    plt.title(f"Route for {ship.name} Ship and Weather selected is {scenario}", fontdict=font1)
    return fig

weather_options = {
    "Calm Seas": [
    [1,1,1,2,2,2,2,1,1,1],
    [1,1,2,2,2,2,2,2,1,1],
    [1,2,2,2,2,2,2,2,2,1],
    [1,2,2,2,2,2,2,2,2,1],
    [1,2,2,2,2,2,2,2,2,1],
    [1,2,2,2,2,2,2,2,2,1],
    [1,2,2,2,2,2,2,2,2,1],
    [1,1,2,2,2,2,2,2,1,1],
    [1,1,1,2,2,2,2,1,1,1],
    [1,1,1,1,2,2,1,1,1,1]
    ],
    "Corner Storm": [
    [1,2,2,2,2,2,2,2,2,2],
    [1,2,2,2,2,2,2,2,2,3],
    [1,2,2,2,2,2,2,2,3,4],
    [1,2,2,2,2,2,2,3,4,5],
    [1,2,2,2,2,2,3,4,5,6],
    [1,2,2,2,2,3,4,5,6,7],
    [1,2,2,2,3,4,5,6,7,8],
    [1,2,2,3,4,5,6,7,8,9],
    [1,2,3,4,5,6,7,8,9,10],
    [2,3,4,5,6,7,8,9,10,10]
    ],
    "Central Storm": [
    [1,1,1,2,2,2,2,1,1,1],
    [1,2,2,3,3,3,3,2,2,1],
    [1,2,3,5,6,6,5,3,2,1],
    [2,3,5,7,8,8,7,5,3,2],
    [2,3,6,8,9,9,8,6,3,2],
    [2,3,6,8,9,9,8,6,3,2],
    [2,3,5,7,8,8,7,5,3,2],
    [1,2,3,5,6,6,5,3,2,1],
    [1,2,2,3,3,3,3,2,2,1],
    [1,1,1,2,2,2,2,1,1,1]
    ],
    "Patchy": [[random.choice([1,2,3,7,8,9]) for j in range(10)] for i in range(10)],
    "Gradient": [
    [1,2,2,3,3,4,5,6,7,8],
    [1,2,3,3,4,5,6,7,8,9],
    [2,3,3,4,5,6,7,8,9,10],
    [2,3,4,5,6,7,8,9,10,10],
    [3,4,5,6,7,8,9,10,10,10],
    [3,4,5,6,7,8,9,10,10,10],
    [4,5,6,7,8,9,10,10,10,10],
    [5,6,7,8,9,10,10,10,10,10],
    [6,7,8,9,10,10,10,10,10,10],
    [7,8,9,10,10,10,10,10,10,10]
    ]
}

ships = {
    "Cargo": Ship("Cargo", speed=15, fuel_rate=2, wave_tolerance=4),
    "Tanker": Ship("Tanker", speed=12, fuel_rate=3, wave_tolerance=5),
    "Passenger": Ship("Passenger", speed=20, fuel_rate=4, wave_tolerance=3)
}

st.title("Ship Routing Simulation on Ocean represented using 10x10 grid")

ship_choice = st.sidebar.selectbox("Select Ship Type", ["Cargo", "Tanker", "Passenger"])
ship = ships[ship_choice]

ship.speed = st.sidebar.number_input("Enter speed (knots)", min_value=1, max_value=30, value=ship.speed)
ship.fuel_rate = st.sidebar.number_input("Enter fuel rate", min_value=1, max_value=10, value=ship.fuel_rate)
ship.wave_tolerance = st.sidebar.slider("Wave tolerance (m)", 1, 10, ship.wave_tolerance)

scenario_choice = st.sidebar.selectbox("Select Weather Scenario", ["Calm Seas", "Corner Storm", "Central Storm", "Patchy", "Gradient"])

weather = weather_options[scenario_choice]


st.header("The Entire path for the given data")
st.subheader("The intensity of blue color represents the weather's condition, the darker it is, the worse it is")

G = build_ocean_grid(width=10, height=10)
start = (0, 0)  
end = (9, 9) 
weather_changes_penalty(G, weather=weather, ship=ship)
path = shortest_path(G, start, end)
distance, time, fuel = compute_stats(path, ship)
st.write(f"**{ship.name} Ship Results**")
st.write(f"Route length: {distance} units")
st.write(f"Travel time: {time:.2f} hours")
st.write("By increasing speed, the ship will take less time to travel")
st.write(f"Fuel used: {fuel} units")
fig = plot_route_with_weather(G, path, ship, scenario_choice)
st.pyplot(fig)

script_dir = Path(__file__).resolve().parent
image_path_obj = script_dir / r"ship.jpg"
image_path = str(image_path_obj)
image_data = mpimg.imread(image_path)


def plot_image_marker(ax, x, y, image_data, zoom=0.075):
    im = OffsetImage(image_data, zoom=zoom)
    ab = AnnotationBbox(im, (x, y), frameon=False, pad=0.0)
    ax.add_artist(ab)

st.header("Animated Simulation")

placeholder = st.empty()
for i in range(1, len(path)+1):
    pos = {(x,y): (x,y) for x,y in G.nodes()}
    node_colors = [weather[x][y] for (x,y) in G.nodes()]
    fig, ax = plt.subplots(figsize=(8,8))
    nx.draw(G, pos, node_size=n_size, node_color=node_colors,
            cmap=plt.cm.Blues, vmin=0, vmax=10, with_labels=False, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=path[:i], node_color="red", ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=list(zip(path[:i-1], path[1:i])),
                           edge_color="red", width=2, ax=ax)
    ax.plot(*pos[path[i-1]], marker="o", markersize=12, color="black")
    plot_image_marker(ax, path[i-1][0], path[i-1][1], image_data=image_data)
    placeholder.pyplot(fig)
    t.sleep(time)

# streamlit run "C:\Users\nafis\OneDrive\Desktop\Ongoing Python Project\Mini_Ship_Routing_System.py"