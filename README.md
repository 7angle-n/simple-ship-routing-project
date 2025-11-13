#**Simple Ship Routing Simulation**

(A mini toy project (also fun project) I made inspired from VISIR-2 ship weather routing model in python)
A dynamic, map-based simulation tool for optimizing ship routes across ocean grids, factoring in weather conditions, ship parameters, and land avoidance. Built with Python, Streamlit, Cartopy, and NetworkX, this project visualizes both static and animated paths for different vessel types under customizable weather scenarios.

###Features
1. Interactive UI with Streamlit for ship selection, weather scenario, and route configuration
2. Real-world map overlay using Cartopy and shapefiles
3. Weather-aware routing with customizable wave matrices and penalties
4. Land avoidance via shapefile-based spatial indexing and STRtree queries
5. Great-circle interpolation for smooth route visualization
6. Animated simulation with ship icons traversing the computed path

####Multiple versions: grid-based abstraction (version 1.1) and real-map routing (version 2.0)

Demo Screenshots
<img width="1919" height="871" alt="Screenshot 2025-11-13 101217" src="https://github.com/user-attachments/assets/58444769-01e2-4e25-adf9-40344ef5cb89" />
<img width="1919" height="880" alt="Screenshot 2025-11-13 101154" src="https://github.com/user-attachments/assets/de4aa4be-15f7-40f6-908d-ec826abf9be0" />
<img width="1919" height="879" alt="Screenshot 2025-11-13 101145" src="https://github.com/user-attachments/assets/a044469c-7813-43d2-94c2-8be541699543" />
<img width="1919" height="860" alt="Screenshot 2025-11-13 101115" src="https://github.com/user-attachments/assets/cb64f08e-8129-43b6-833b-4a05ed2bbc29" />
<img width="1919" height="837" alt="Screenshot 2025-11-13 101038" src="https://github.com/user-attachments/assets/ab82179b-514d-464b-9d34-f77e40319764" />


###Inputs & Controls
1. Ship Type: Cargo, Tanker, Passenger
2. Speed, Fuel Rate, Wave Tolerance: Adjustable per vessel
3. Weather Scenarios: Calm Seas, Central Storm, Gradient, Patchy, etc.
4. Start/End Coordinates: Custom lat/lon inputs
5. Fuel Price, Beam, Direction: Optional economic and geometric parameters

###Outputs
1. Route Distance: In nautical miles
2. Travel Time: Based on ship speed
3. Fuel Penalty: Based on weather and ship tolerance
4. Static Map: With wave height heatmap and route overlay
5. Animated Simulation: Ship traversing the route frame-by-frame

###Technical Highlights
1. Graph Construction: Ocean-only nodes with land avoidance
2. Shortest Path: Dijkstra’s algorithm with weather-weighted edges
3. Interpolation: Great-circle segments for realism
4. Caching: Streamlit resource caching for shapefile and graph loading
5. Visualization: Cartopy for geospatial plotting, Matplotlib for animation
