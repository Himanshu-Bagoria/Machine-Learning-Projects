import streamlit as st
import requests
import random
import pandas as pd
from datetime import datetime
import plotly.express as px
import folium
from streamlit_folium import st_folium

# --- Page Config ---
st.set_page_config(page_title=" Live Weather Forecast", page_icon="🌤️", layout="wide")

# --- Initialize Session State ---
if 'page' not in st.session_state:
    st.session_state.page = "Home"
if 'selected_city' not in st.session_state:
    st.session_state.selected_city = "Delhi"

# --- Constants ---
API_KEY = "4a15567c64817ac9900efd1c6ec21144"
CITIES = {
    "Delhi": [28.6139, 77.2090],
    "Mumbai": [19.0760, 72.8777],
    "London": [51.5074, -0.1278],
    "New York": [40.7128, -74.0060],
    "Tokyo": [35.6895, 139.6917],
    "Dubai": [25.2048, 55.2708],
    "Paris": [48.8566, 2.3522],
    "Sydney": [-33.8688, 151.2093],
    "Singapore": [1.3521, 103.8198],
    "Kolkata": [22.5726, 88.3639],
    "Chennai": [13.0827, 80.2707],
    "Bengaluru": [12.9716, 77.5946],
    "Hyderabad": [17.3850, 78.4867],
    "Pune": [18.5204, 73.8567],
    "Ahmedabad": [23.0225, 72.5714],
    "Jaipur": [26.9124, 75.7873],
    "Lucknow": [26.8467, 80.9462],
    "Bhopal": [23.2599, 77.4126],
    "Patna": [25.5941, 85.1376],
    "Chandigarh": [30.7333, 76.7794],
    "Indore": [22.7196, 75.8577],
    "Nagpur": [21.1458, 79.0882],
    "Surat": [21.1702, 72.8311],
    "Kanpur": [26.4499, 80.3319],
    "Coimbatore": [11.0168, 76.9558],
    "Thiruvananthapuram": [8.5241, 76.9366],
    "Guwahati": [26.1445, 91.7362],
    "Ranchi": [23.3441, 85.3096],
    "Bhubaneswar": [20.2961, 85.8245],
    "Mysuru": [12.2958, 76.6394],
    "Varanasi": [25.3176, 82.9739]
}

# --- Load CSS ---
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- Helper Functions ---
def format_time_12h(ts):
    return datetime.fromtimestamp(ts).strftime('%I:%M %p')

def get_weather_data(city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    return requests.get(url).json()

def get_forecast_data(city):
    url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={API_KEY}&units=metric"
    return requests.get(url).json()

# --- Animation Renderers ---
def render_rain():
    drops = "".join([f'<div class="drop" style="left: {random.randint(0, 100)}%; animation-delay: {random.uniform(0, 2)}s;"></div>' for _ in range(50)])
    st.markdown(f'<div class="rain">{drops}</div>', unsafe_allow_html=True)

def render_stars():
    stars = '<div class="moon"></div>'
    for _ in range(80):
        left, top, size, duration = random.randint(0, 100), random.randint(0, 100), random.randint(1, 3), random.uniform(1, 4)
        stars += f'<div class="star" style="left: {left}%; top: {top}%; width: {size}px; height: {size}px; --duration: {duration}s;"></div>'
    st.markdown(f'<div class="stars">{stars}</div>', unsafe_allow_html=True)

def render_sun():
    st.markdown('<div class="sun"></div>', unsafe_allow_html=True)

def render_clouds():
    clouds = "".join([f'<div class="cloud" style="top: {random.randint(5, 25)}%; animation-delay: -{random.randint(0, 10)}s; animation-duration: {random.randint(15, 30)}s;"></div>' for _ in range(5)])
    st.markdown(f'<div class="clouds">{clouds}</div>', unsafe_allow_html=True)

def get_bg_style(bg_class):
    styles = {
        "sunny-bg": "linear-gradient(135deg, #333333 0%, #1a1a1a 100%)",
        "night-bg": "linear-gradient(135deg, #1f1f1f 0%, #0a0a0a 100%)",
        "rainy-bg": "linear-gradient(135deg, #2b2b2b 0%, #121212 100%)",
        "cloudy-bg": "linear-gradient(135deg, #3a3a3a 0%, #202020 100%)"
    }
    return f"<style>.stApp {{ background: {styles.get(bg_class, styles['sunny-bg'])}; color: white; }}</style>"

# --- UI Components ---
def render_top_navbar():
    # Container for layout
    nav_cols = st.columns([5, 1, 1, 1])
    with nav_cols[1]:
        if st.button("🏠 Home", use_container_width=True):
            st.session_state.page = "Home"
            st.rerun()
    with nav_cols[2]:
        if st.button("🗺️ Map", use_container_width=True):
            st.session_state.page = "Map"
            st.rerun()
    with nav_cols[3]:
        if st.button("📊 Compare", use_container_width=True):
            st.session_state.page = "Compare"
            st.rerun()

# --- Page Renderers ---
def show_home():
    st.markdown("<h1 class='main-title'>✨ Live Weather Forecast</h1>", unsafe_allow_html=True)
    
    # Logic to handle selected city from Map
    cities_list = list(CITIES.keys())
    selected_city = st.selectbox("📍 Select Your Destination", cities_list, index=cities_list.index(st.session_state.selected_city))
    st.session_state.selected_city = selected_city

    curr_res = get_weather_data(selected_city)
    fore_res = get_forecast_data(selected_city)

    if "main" in curr_res:
        # Visual State Logic
        temp, humidity, wind = curr_res['main']['temp'], curr_res['main']['humidity'], curr_res['wind']['speed']
        condition, desc = curr_res['weather'][0]['main'], curr_res['weather'][0]['description'].capitalize()
        icon_code = curr_res['weather'][0]['icon']
        is_night = 'n' in icon_code
        
        bg_class = "sunny-bg"
        if any(x in condition for x in ["Rain", "Drizzle", "Thunderstorm"]):
            bg_class = "rainy-bg"; render_rain()
        elif is_night:
            bg_class = "night-bg"; render_stars()
        elif "Clouds" in condition:
            bg_class = "cloudy-bg"; render_clouds()
        elif condition == "Clear":
            bg_class = "sunny-bg"; render_sun()
        else:
            bg_class = "sunny-bg"
            
        st.markdown(get_bg_style(bg_class), unsafe_allow_html=True)

        # 1. Header Card (HTML)
        header_html = f"""<div class="weather-card-container"><div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap;"><div style="flex: 1; min-width: 200px;"><div style='font-size: 1.2rem; opacity: 0.8;'>{selected_city}</div><div style='display: flex; align-items: center;'><img src='http://openweathermap.org/img/wn/{icon_code}@4x.png' width='80'><span style='font-size: 4.5rem; font-weight: 600;'>{int(temp)}°C</span></div><div style='font-size: 1.1rem; text-transform: uppercase; letter-spacing: 2px;'>{condition}</div><div style='font-size: 0.85rem; opacity: 0.7;'>{desc}</div></div><div style='flex: 1; min-width: 180px; background: rgba(255,255,255,0.1); padding: 15px; border-radius: 15px; margin-top: 10px;'><div style='font-size: 0.85rem; margin-bottom: 8px;'>Precipitation: <b>{curr_res.get('rain', {}).get('1h', 0)}%</b></div><div style='font-size: 0.85rem; margin-bottom: 8px;'>Humidity: <b>{humidity}%</b></div><div style='font-size: 0.85rem; margin-bottom: 8px;'>Wind: <b>{wind} m/s</b></div><div style='font-size: 0.85rem;'>Feels Like: <b>{curr_res['main']['feels_like']}°C</b></div></div></div></div>"""
        st.markdown(header_html, unsafe_allow_html=True)

        # 2. Temperature Trend Chart
        if "list" in fore_res:
            with st.container():
                st.markdown("<div class='weather-card-container'><div style='font-size: 1rem; margin-bottom: 5px; opacity: 0.9;'>Temperature Trend (Next 24h - 12hr format)</div>", unsafe_allow_html=True)
                df = pd.DataFrame([{
                    'Time': format_time_12h(f['dt']), 
                    'Temp': f['main']['temp'],
                    'Condition': f['weather'][0]['main']
                } for f in fore_res['list'][:8]])
                fig = px.bar(df, x='Time', y='Temp', color='Temp', color_continuous_scale='YlOrRd', 
                             hover_data={'Condition': True, 'Temp': ':.1f', 'Time': True},
                             labels={'Temp': 'Temp (°C)', 'Time': 'Time'}, text='Condition')
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white", 
                                  margin=dict(l=0, r=0, t=30, b=0), height=300, xaxis_title=None, yaxis_title=None, coloraxis_showscale=False)
                fig.update_traces(marker_line_color='rgba(0,0,0,0)', textposition='outside', textfont=dict(size=10))
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                st.markdown("</div>", unsafe_allow_html=True)

        # 3. 5-Day Forecast
        forecast_html = "<div class='weather-card-container'><div class='forecast-container'>"
        seen_days = set()
        for f in fore_res.get('list', []):
            day = datetime.fromtimestamp(f['dt']).strftime('%a')
            if day not in seen_days and len(seen_days) < 5:
                seen_days.add(day)
                f_icon, f_temp = f['weather'][0]['icon'], int(f['main']['temp'])
                f_main = f['weather'][0]['main']
                forecast_html += f'<div class="forecast-mini-card"><div class="forecast-day">{day}</div><img src="http://openweathermap.org/img/wn/{f_icon}.png" width="35"><div class="forecast-temp">{f_temp}°C</div><div class="forecast-desc">{f_main}</div></div>'
        forecast_html += "</div></div>"
        st.markdown(forecast_html, unsafe_allow_html=True)
    else:
        st.error("Connect error or city not found.")

def show_map():
    st.markdown("<h1 class='main-title'>🌍 Live Weather Map</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; opacity: 0.8;'>Explore global conditions dynamically. Click markers for details.</p>", unsafe_allow_html=True)
    
    # Create dark-themed map
    m = folium.Map(location=[20, 0], zoom_start=2, tiles="CartoDB dark_matter")
    
    for city, coords in CITIES.items():
        try:
            res = get_weather_data(city)
            temp = int(res['main']['temp'])
            cond = res['weather'][0]['main']
            
            # Popup with "More Details" logic
            popup_html = f"""
                <div style="color: #333; font-family: sans-serif; width: 140px; padding: 5px;">
                    <h5 style="margin: 0 0 5px 0; color: #1e293b;">{city}</h5>
                    <p style="margin: 0; font-size: 0.9rem;"><b>{temp}°C</b> | {cond}</p>
                    <hr style="margin: 8px 0;">
                    <p style="margin: 0; font-size: 0.8rem; color: #64748b;">(Click marker label to view full dashboard)</p>
                </div>
            """
            folium.Marker(
                location=coords,
                popup=folium.Popup(popup_html, max_width=200),
                tooltip=city,
                icon=folium.Icon(color="orange" if temp > 25 else "blue", icon="info-sign")
            ).add_to(m)
        except: continue

    st_map = st_folium(m, width="100%", height=600)
    
    # If a marker is clicked (based on tooltip), redirect to Home
    if st_map.get("last_object_clicked_tooltip"):
        st.session_state.selected_city = st_map["last_object_clicked_tooltip"]
        st.session_state.page = "Home"
        st.rerun()

def show_compare():
    st.markdown("<h1 class='main-title'>📊 City Comparison</h1>", unsafe_allow_html=True)
    
    selected_compare = st.multiselect("Select Cities to Compare (Up to 3)", list(CITIES.keys()), default=["Delhi", "Mumbai"])
    
    if selected_compare:
        cols = st.columns(len(selected_compare))
        for i, city in enumerate(selected_compare[:3]):
            with cols[i]:
                res = get_weather_data(city)
                if "main" in res:
                    temp = int(res['main']['temp'])
                    cond = res['weather'][0]['main']
                    hum = res['main']['humidity']
                    icon = res['weather'][0]['icon']
                    
                    st.markdown(f"""
                        <div class="compare-card">
                            <h3 style="margin: 0;">{city}</h3>
                            <img src="http://openweathermap.org/img/wn/{icon}@2x.png" width="100">
                            <div style="font-size: 3rem; font-weight: bold; margin: 10px 0;">{temp}°C</div>
                            <div style="text-transform: uppercase; letter-spacing: 2px; font-weight: 500;">{cond}</div>
                            <hr style="opacity: 0.2; margin: 15px 0;">
                            <div style="font-size: 0.9rem; opacity: 0.7;">Humidity: {hum}%</div>
                            <div style="font-size: 0.9rem; opacity: 0.7;">Wind: {res['wind']['speed']} m/s</div>
                        </div>
                    """, unsafe_allow_html=True)

# --- Main App Execution ---
render_top_navbar()

if st.session_state.page == "Home":
    show_home()
elif st.session_state.page == "Map":
    show_map()
elif st.session_state.page == "Compare":
    show_compare()

st.markdown("<br><p style='text-align: center; color: #94a3b8; font-size: 0.8rem; margin-top: 50px;'>Powered by OpenWeatherMap API • Redesigned with ✨</p>", unsafe_allow_html=True)
