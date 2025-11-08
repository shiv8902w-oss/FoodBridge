# main.py
import streamlit as st
import pandas as pd
from streamlit_folium import st_folium
import folium
from geopy.distance import geodesic
from sklearn.cluster import KMeans
import numpy as np
import math

st.set_page_config(page_title="FoodBridge - Map & Prediction", layout="wide")
st.title("🌍 FoodBridge — Donor ↔ Needy Matching & Demand Simulation")

@st.cache_data
def load_data(donor_file="donor_india.csv", needy_file="poor_dataset.csv"):
    donors = pd.read_csv(donor_file)
    needy = pd.read_csv(needy_file)
    return donors, needy

# Load
donor_file = st.sidebar.text_input("Donor CSV filename", value="donor_india.csv")
needy_file = st.sidebar.text_input("Needy CSV filename", value="poor_dataset.csv")

try:
    donors, needy = load_data(donor_file, needy_file)
except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()

# Validate lat/lon
for df, name in [(donors, "Donors"), (needy, "Needy")]:
    if not {'latitude', 'longitude'}.issubset(df.columns):
        st.error(f"{name} dataset must contain 'latitude' and 'longitude' columns.")
        st.stop()

donors = donors.dropna(subset=['latitude', 'longitude']).reset_index(drop=True)
needy = needy.dropna(subset=['latitude', 'longitude']).reset_index(drop=True)

# Top metrics
col1, col2, col3, col4 = st.columns([1,1,1,1])
col1.metric("🟢 Total Donors", len(donors))
col2.metric("🔴 Total Needy", len(needy))
# compute avg nearest distance if possible
def avg_nearest_distance(donors, needy, sample=200):
    if donors.empty or needy.empty:
        return None
    # sample to speed up
    ns = needy if len(needy) <= sample else needy.sample(sample, random_state=1)
    total = 0.0
    for _, nr in ns.iterrows():
        p = (nr['latitude'], nr['longitude'])
        # compute donors distances quickly
        dists = donors.apply(lambda r: geodesic(p, (r['latitude'], r['longitude'])).km, axis=1)
        total += dists.min()
    return total / len(ns)

avgd = avg_nearest_distance(donors, needy)
col3.metric("📏 Avg nearest dist (km)", f"{avgd:.2f}" if avgd else "N/A")

col4.metric("🤝 Estimated matches", min(len(donors), len(needy)))

st.markdown("---")

# Controls
with st.sidebar.expander("Matching & Prediction Controls", expanded=True):
    st.write("Nearest matching and demand-cluster simulation")
    do_match = st.checkbox("Show nearest-donor connections", value=True)
    do_cluster = st.checkbox("Show demand hotspots (KMeans)", value=True)
    n_clusters = st.slider("Number of clusters for hotspots", 3, 25, 7)
    growth_rate = st.slider("Simulated growth rate for prediction (%)", 0, 200, 20)
    max_lines = st.number_input("Max match lines to draw (for performance)", min_value=10, max_value=1000, value=200)
    show_popups = st.checkbox("Show popup info on lines", value=True)

# Create base map centered on India
m = folium.Map(location=[22.9734, 78.6569], zoom_start=5, tiles="OpenStreetMap")

# Custom icons
donor_icon_url = "https://maps.google.com/mapfiles/ms/icons/green-dot.png"
needy_icon_url = "https://maps.google.com/mapfiles/ms/icons/red-dot.png"

# Add donor markers
donor_group = folium.FeatureGroup(name="Donors", show=True)
for idx, r in donors.iterrows():
    popup_html = f"<b>Donor:</b> {r.get('name', 'Unknown')}<br>"
    if 'capacity' in r:
        popup_html += f"<b>Capacity:</b> {r['capacity']}<br>"
    popup_html += f"<i>coords:</i> {r['latitude']:.4f}, {r['longitude']:.4f}"
    folium.CircleMarker(
        location=[r['latitude'], r['longitude']],
        radius=5,
        color="green",
        fill=True,
        fill_color="green",
        popup=folium.Popup(popup_html, max_width=250)
    ).add_to(donor_group)
donor_group.add_to(m)

# Add needy markers
needy_group = folium.FeatureGroup(name="Needy", show=True)
for idx, r in needy.iterrows():
    popup_html = f"<b>Needy:</b> {r.get('name', 'Unknown')}<br>"
    if 'need' in r:
        popup_html += f"<b>Need:</b> {r['need']}<br>"
    popup_html += f"<i>coords:</i> {r['latitude']:.4f}, {r['longitude']:.4f}"
    folium.CircleMarker(
        location=[r['latitude'], r['longitude']],
        radius=5,
        color="red",
        fill=True,
        fill_color="red",
        popup=folium.Popup(popup_html, max_width=250)
    ).add_to(needy_group)
needy_group.add_to(m)

# Matching: for each needy, find nearest donor and draw polyline
if do_match and (not donors.empty) and (not needy.empty):
    st.info("Computing nearest donor for each needy (this may take a few seconds).")
    # Convert donors coords to list to speed up
    donor_coords = donors[['latitude', 'longitude']].values.tolist()
    # We'll store matches
    matches = []
    # iterate needy (limit lines drawn to max_lines to avoid huge maps)
    for i, nr in needy.iterrows():
        p = (nr['latitude'], nr['longitude'])
        # compute distances to all donors (vectorized loop)
        best_idx = None
        best_dist = float('inf')
        for j, dcoord in enumerate(donor_coords):
            dist_km = geodesic(p, (dcoord[0], dcoord[1])).km
            if dist_km < best_dist:
                best_dist = dist_km
                best_idx = j
        if best_idx is not None:
            donor_row = donors.iloc[best_idx]
            matches.append({
                'needy_index': i,
                'donor_index': best_idx,
                'needy_coord': p,
                'donor_coord': (donor_row['latitude'], donor_row['longitude']),
                'distance_km': best_dist,
                'donor_name': donor_row.get('name', 'Donor'),
                'needy_name': nr.get('name', 'Needy')
            })
    # Sort matches by distance and optionally limit how many lines to draw
    matches = sorted(matches, key=lambda x: x['distance_km'])
    drawn = 0
    for match in matches:
        if drawn >= max_lines:
            break
        coords = [match['donor_coord'], match['needy_coord']]
        # blue line for connection
        line = folium.PolyLine(locations=coords, color="blue", weight=2, opacity=0.6)
        line.add_to(m)
        if show_popups:
            folium.Popup(f"Donor: {match['donor_name']}<br>Needy: {match['needy_name']}<br>Distance: {match['distance_km']:.2f} km",
                         max_width=300).add_to(line)
        drawn += 1
    st.success(f"Plotted {min(len(matches), max_lines)} nearest connections.")

# Clustering & Demand Prediction (simple simulation)
if do_cluster and (not needy.empty):
    st.info("Clustering needy points to find demand hotspots...")
    coords = needy[['latitude', 'longitude']].values
    # KMeans can fail if too few points relative to clusters
    n_clusters_use = min(n_clusters, len(coords))
    if n_clusters_use < 1:
        st.warning("Not enough needy points for clustering.")
    else:
        kmeans = KMeans(n_clusters=n_clusters_use, random_state=0).fit(coords)
        centers = kmeans.cluster_centers_
        labels = kmeans.labels_
        needy['cluster'] = labels
        # compute cluster counts -> current demand
        cluster_counts = needy.groupby('cluster').size().to_dict()
        # simulate predicted demand via growth_rate
        predicted = {c: int(math.ceil(cluster_counts.get(c, 0) * (1 + growth_rate / 100.0))) for c in range(n_clusters_use)}

        # draw cluster centers and circle sizes proportional to predicted demand
        for c_idx, center in enumerate(centers):
            latc, lonc = center[0], center[1]
            curr = cluster_counts.get(c_idx, 0)
            pred = predicted[c_idx]
            # radius in meters: scale base on predicted count
            radius = 200 + pred * 30  # tweak factor for visualization
            color = "#ff4d4d"
            folium.Circle(
                location=(latc, lonc),
                radius=radius,
                color=color,
                fill=True,
                fill_opacity=0.25,
                tooltip=f"Hotspot {c_idx}: current={curr}, predicted={pred}"
            ).add_to(m)
            # label center with predicted value
            folium.map.Marker(
                [latc, lonc],
                icon=folium.DivIcon(html=f"""<div style="font-size:12px;color:#fff;
                    background:#ff4d4d;padding:4px 6px;border-radius:6px">
                    🔥 {pred}</div>""")
            ).add_to(m)
        st.success(f"Created {n_clusters_use} hotspots with predicted demand (growth rate {growth_rate}%).")

# Add layer control
folium.LayerControl().add_to(m)

# Draw map
st.subheader("Map View — Donors (green), Needy (red), matches (blue lines), hotspots (red circles)")
st_data = st_folium(m, width=1100, height=650)

# Analytics panel below map
st.markdown("---")
st.header("Analytics & Next Steps")
st.markdown("""
- *Nearest-match lines* show a greedy matching approach (nearest donor assigned to each needy).
- *Hotspot clustering + growth slider* simulates demand prediction. Replace growth slider with a trained model output for real AI forecasts.
- Next steps: perform capacity-aware matching (donor capacity vs cluster demand), route optimization for volunteers, and integrate real-time data.
""")

# Show a table of closest matches (top 20)
if do_match and len(matches) > 0:
    st.subheader("Closest Matches (Top 20)")
    df_matches = pd.DataFrame(matches)[:20]
    df_matches_display = df_matches[['donor_name', 'needy_name', 'distance_km']]
    st.dataframe(df_matches_display)

# Option to download matches as CSV
if do_match and len(matches) > 0:
    export_df = pd.DataFrame(matches)
    csv = export_df.to_csv(index=False).encode('utf-8')
    st.download_button(label="Download matches CSV", data=csv, file_name='matches.csv', mime='text/csv')