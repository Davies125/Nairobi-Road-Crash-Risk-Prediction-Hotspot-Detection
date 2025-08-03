import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, time
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import time as time_module
import warnings
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist, squareform
import io
warnings.filterwarnings('ignore')

# Configure the page
st.set_page_config(
    page_title="Nairobi Crash Risk Predictor & Hotspot Analysis",
    page_icon="🚗",
    layout="wide"
)

@st.cache_data
def load_data():
    """Load the crash data"""
    try:
        df = pd.read_excel('nairobi-road-crashes-data_public-copy.xlsx', 
                          sheet_name='ma3route_crashes_algorithmcode')
        return df
    except FileNotFoundError:
        st.error("❌ Dataset file not found! Please add 'nairobi-road-crashes-data_public-copy.xlsx' to the app folder.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None

@st.cache_resource
def load_models():
    """Load the best trained model"""
    try:
        model = joblib.load('model_xgboost.pkl')
        label_encoder = joblib.load('label_encoder.pkl')
        return model, label_encoder
    except FileNotFoundError as e:
        st.warning(f"⚠️ Model files not found. Please ensure 'model_xgboost.pkl' and 'label_encoder.pkl' are in the app directory.")
        return None, None
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        return None, None

def preprocess_data(df):
    """Preprocess the data like in training"""
    if df is None:
        return None
        
    df['crash_datetime'] = pd.to_datetime(df['crash_datetime'])
    df['crash_date'] = pd.to_datetime(df['crash_date']).dt.date
    
    df['crash_hour'] = df['crash_datetime'].dt.hour
    df['crash_dayofweek'] = df['crash_datetime'].dt.dayofweek
    df['crash_month'] = df['crash_datetime'].dt.month
    
    df['fatal_crash'] = df['contains_fatality_words'].astype(int)
    df['pedestrian_involved'] = df['contains_pedestrian_words'].astype(int)
    df['matatu_involved'] = df['contains_matatu_words'].astype(int)
    df['motorcycle_involved'] = df['contains_motorcycle_words'].astype(int)
    
    df['rush_hour'] = df['crash_hour'].apply(lambda x: 1 if (7 <= x <= 9) or (16 <= x <= 19) else 0)
    df['weekend'] = df['crash_dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
    df['night_time'] = df['crash_hour'].apply(lambda x: 1 if x >= 22 or x <= 5 else 0)
    
    df['severity_score'] = (df['fatal_crash'] * 3 + 
                           df['pedestrian_involved'] * 2 + 
                           df['matatu_involved'] * 1.5 + 
                           df['motorcycle_involved'] * 1.2)
    
    df['severity_category'] = pd.cut(df['severity_score'], 
                                    bins=[-0.1, 0, 1, 2, 10], 
                                    labels=['No_Injury', 'Minor', 'Moderate', 'Severe'])
    
    return df

def detect_crash_hotspots(df, eps_km=0.5, min_samples=5):
    """Detect crash hotspots using DBSCAN clustering"""
    if df is None or len(df) == 0:
        return pd.DataFrame()
    
    valid_coords = df.dropna(subset=['latitude', 'longitude'])
    if len(valid_coords) < min_samples:
        return pd.DataFrame()
    
    coords_rad = np.radians(valid_coords[['latitude', 'longitude']].values)
    eps_rad = eps_km / 6371.0
    clustering = DBSCAN(eps=eps_rad, min_samples=min_samples, metric='haversine')
    clusters = clustering.fit_predict(coords_rad)
    
    valid_coords = valid_coords.copy()
    valid_coords['cluster'] = clusters
    
    hotspots = []
    for cluster_id in set(clusters):
        if cluster_id != -1:
            cluster_data = valid_coords[valid_coords['cluster'] == cluster_id]
            hotspot = {
                'cluster_id': cluster_id,
                'center_lat': cluster_data['latitude'].mean(),
                'center_lon': cluster_data['longitude'].mean(),
                'crash_count': len(cluster_data),
                'severe_count': len(cluster_data[cluster_data['severity_category'] == 'Severe']),
                'risk_score': len(cluster_data[cluster_data['severity_category'] == 'Severe']) / len(cluster_data) if len(cluster_data) > 0 else 0
            }
            hotspots.append(hotspot)
    
    return pd.DataFrame(hotspots).sort_values('risk_score', ascending=False)

def calculate_dynamic_risk_score(severity, confidence, nearby_crashes, weather_factor=1.0, traffic_factor=1.0):
    """Calculate dynamic risk score with multiple factors"""
    base_scores = {'No_Injury': 1, 'Minor': 2, 'Moderate': 3, 'Severe': 4}
    base_score = base_scores.get(severity, 1)
    
    nearby_severe = len(nearby_crashes[nearby_crashes['severity_category'] == 'Severe']) if len(nearby_crashes) > 0 else 0
    nearby_factor = 1 + (nearby_severe * 0.1)
    
    confidence_factor = confidence
    
    risk_score = min(10, base_score * confidence_factor * nearby_factor * weather_factor * traffic_factor)
    
    return risk_score

def get_risk_level(risk_score):
    """Convert risk score to risk level"""
    if risk_score >= 7:
        return "EXTREME RISK", "red"
    elif risk_score >= 5:
        return "HIGH RISK", "orange"
    elif risk_score >= 3:
        return "MODERATE RISK", "yellow"
    else:
        return "LOW RISK", "green"

def get_traffic_risk_factor(hour, day_of_week):
    """Get traffic-based risk factor"""
    if (7 <= hour <= 9) or (16 <= hour <= 19):
        return 1.3
    elif 22 <= hour or hour <= 5:
        return 0.8
    else:
        return 1.0

def get_coordinates_from_location(location_name):
    """Get coordinates from location name using geocoding"""
    try:
        geolocator = Nominatim(user_agent="nairobi_crash_app")
        full_location = f"{location_name}, Nairobi, Kenya"
        location = geolocator.geocode(full_location, timeout=10)
        
        if location:
            return location.latitude, location.longitude, location.address
        else:
            return None, None, None
    except GeocoderTimedOut:
        return None, None, None
    except Exception as e:
        st.error(f"Geocoding error: {str(e)}")
        return None, None, None

def get_nearby_crashes(df, lat, lon, radius_km=2):
    """Get crashes within radius of given coordinates"""
    if df is None:
        return pd.DataFrame()
    
    valid_coords = df.dropna(subset=['latitude', 'longitude'])
    if len(valid_coords) == 0:
        return pd.DataFrame()
    
    earth_radius = 6371
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    
    crash_lat_rad = np.radians(valid_coords['latitude'])
    crash_lon_rad = np.radians(valid_coords['longitude'])
    
    dlat = crash_lat_rad - lat_rad
    dlon = crash_lon_rad - lon_rad
    
    a = np.sin(dlat/2)**2 + np.cos(lat_rad) * np.cos(crash_lat_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    distances = earth_radius * c
    
    nearby_mask = distances <= radius_km
    nearby_crashes = valid_coords[nearby_mask].copy()
    nearby_crashes['distance_km'] = distances[nearby_mask]
    
    return nearby_crashes.sort_values('distance_km')

def get_safety_recommendations(severity, hour, is_weekend, is_rush_hour, is_night, nearby_crashes):
    """Generate safety recommendations based on prediction and context"""
    recommendations = []
    
    if severity == 'Severe':
        recommendations.extend([
            "🚨 HIGH RISK AREA - Exercise extreme caution",
            "🚗 Reduce speed significantly below speed limit",
            "👀 Maintain extra vigilance for pedestrians and other vehicles"
        ])
    elif severity == 'Moderate':
        recommendations.extend([
            "⚠️ MODERATE RISK - Stay alert",
            "🚗 Drive defensively and maintain safe following distance"
        ])
    else:
        recommendations.append("✅ Relatively safe area - maintain normal precautions")
    
    if is_rush_hour:
        recommendations.append("🚦 Rush hour traffic - expect congestion and aggressive driving")
    
    if is_night:
        recommendations.extend([
            "🌙 Night time driving - use headlights and reduce speed",
            "👁️ Extra vigilance for pedestrians and cyclists",
            "🔦 Ensure good visibility and avoid fatigue"
        ])
    
    if is_weekend:
        recommendations.append("🍺 Weekend - watch for impaired drivers, especially at night")
    
    if 6 <= hour <= 8:
        recommendations.append("⏰ School hours - watch for children near schools")
    
    if len(nearby_crashes) > 0:
        severe_nearby = len(nearby_crashes[nearby_crashes['severity_category'] == 'Severe'])
        if severe_nearby > 5:
            recommendations.append(f"📍 {severe_nearby} severe crashes recorded nearby - known danger zone")
        
        if nearby_crashes['pedestrian_involved'].sum() > 3:
            recommendations.append("🚶 High pedestrian accident area - reduce speed near crossings")
        if nearby_crashes['matatu_involved'].sum() > 3:
            recommendations.append("🚌 High matatu accident area - maintain distance from public transport")
        if nearby_crashes['motorcycle_involved'].sum() > 3:
            recommendations.append("🏍️ High motorcycle accident area - check blind spots carefully")
    
    return recommendations

def generate_risk_report(prediction_results):
    """Generate downloadable risk assessment report"""
    risk_score = calculate_dynamic_risk_score(
        prediction_results['severity'],
        prediction_results['confidence'],
        prediction_results['nearby_crashes']
    )
    risk_level, _ = get_risk_level(risk_score)
    
    report = f"""NAIROBI CRASH RISK ASSESSMENT REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

LOCATION DETAILS:
- Address: {prediction_results['location']['address']}
- Coordinates: {prediction_results['location']['lat']:.6f}, {prediction_results['location']['lon']:.6f}
- Date/Time: {prediction_results['datetime']['date']} at {prediction_results['datetime']['time']}

RISK ASSESSMENT:
- Predicted Severity: {prediction_results['severity']}
- Confidence Level: {prediction_results['confidence']:.1%}
- Dynamic Risk Score: {risk_score:.1f}/10
- Risk Level: {risk_level}

CONTEXTUAL FACTORS:
- Rush Hour: {'Yes' if prediction_results['datetime']['rush_hour'] else 'No'}
- Weekend: {'Yes' if prediction_results['datetime']['weekend'] else 'No'}
- Night Time: {'Yes' if prediction_results['datetime']['night_time'] else 'No'}
- Nearby Crashes (2km): {len(prediction_results['nearby_crashes'])}

SAFETY RECOMMENDATIONS:
"""
    for i, rec in enumerate(prediction_results['recommendations'], 1):
        report += f"{i}. {rec.replace('🚨', '').replace('⚠️', '').replace('✅', '').replace('🚦', '').replace('⏰', '').replace('🚌', '').replace('🌙', '').replace('👁️', '').replace('🔦', '').replace('🍺', '').replace('📍', '').replace('🚶', '').replace('🏍️', '').strip()}\n"
    
    report += f"""
DISCLAIMER:
This risk assessment is based on historical crash data and machine learning predictions. 
It should be used as a guide only and does not guarantee safety outcomes. 
Always follow traffic rules and drive according to current conditions.

Report generated by Nairobi Crash Risk Predictor v2.0
"""
    return report

def save_to_history(prediction_results):
    """Save prediction to user history"""
    if 'user_history' not in st.session_state:
        st.session_state.user_history = []
    
    st.session_state.user_history.append({
        'timestamp': datetime.now(),
        'location': prediction_results['location']['address'],
        'severity': prediction_results['severity'],
        'confidence': prediction_results['confidence'],
        'risk_score': calculate_dynamic_risk_score(
            prediction_results['severity'],
            prediction_results['confidence'],
            prediction_results['nearby_crashes']
        )
    })
    
    # Keep only last 50 predictions
    if len(st.session_state.user_history) > 50:
        st.session_state.user_history = st.session_state.user_history[-50:]

def calculate_route_risk(df, start_lat, start_lon, end_lat, end_lon, num_points=10):
    """Calculate risk along a route by sampling points"""
    route_points = []
    
    # Generate points along the route (simple linear interpolation)
    for i in range(num_points):
        t = i / (num_points - 1)
        lat = start_lat + t * (end_lat - start_lat)
        lon = start_lon + t * (end_lon - start_lon)
        route_points.append((lat, lon))
    
    route_risks = []
    for lat, lon in route_points:
        nearby_crashes = get_nearby_crashes(df, lat, lon, radius_km=1)
        if len(nearby_crashes) > 0:
            severe_count = len(nearby_crashes[nearby_crashes['severity_category'] == 'Severe'])
            risk = severe_count / len(nearby_crashes) if len(nearby_crashes) > 0 else 0
        else:
            risk = 0
        route_risks.append(risk)
    
    return route_points, route_risks

# Comprehensive Nairobi locations for quick selection
NAIROBI_LOCATIONS = {
    # Central Business District & City Center
    "CBD/City Center": (-1.2864, 36.8172),
    "Kenyatta Avenue": (-1.2833, 36.8167),
    "Uhuru Highway": (-1.2864, 36.8172),
    "Haile Selassie Avenue": (-1.2889, 36.8236),
    "Tom Mboya Street": (-1.2833, 36.8194),
    "Moi Avenue": (-1.2847, 36.8208),
    "Ronald Ngala Street": (-1.2819, 36.8181),
    "River Road": (-1.2806, 36.8236),
    "Latema Road": (-1.2792, 36.8222),
    "Accra Road": (-1.2875, 36.8153),
    
    # Westlands & Surroundings
    "Westlands": (-1.2676, 36.8108),
    "Sarit Centre": (-1.2639, 36.8083),
    "ABC Place": (-1.2653, 36.8097),
    "Westgate Mall": (-1.2653, 36.8097),
    "Chiromo": (-1.2708, 36.8056),
    "Parklands": (-1.2630, 36.8581),
    "Highridge": (-1.2597, 36.8042),
    "Spring Valley": (-1.2542, 36.8000),
    
    # Kilimani & Surroundings
    "Kilimani": (-1.2921, 36.7872),
    "Yaya Centre": (-1.2931, 36.7881),
    "Hurlingham": (-1.2958, 36.7833),
    "Kileleshwa": (-1.2836, 36.7672),
    "Lavington": (-1.2836, 36.7672),
    "Dennis Pritt Road": (-1.2958, 36.7806),
    "Wood Avenue": (-1.2944, 36.7889),
    "Argwings Kodhek Road": (-1.2931, 36.7847),
    
    # Karen & Langata
    "Karen": (-1.3197, 36.7085),
    "Karen Shopping Centre": (-1.3181, 36.7097),
    "Langata": (-1.3515, 36.7519),
    "Langata Road": (-1.3364, 36.7519),
    "Wilson Airport": (-1.3208, 36.8153),
    "Nairobi National Park Gate": (-1.3736, 36.8583),
    "Galleria Mall": (-1.3181, 36.7097),
    "Junction Mall": (-1.3208, 36.7125),
    
    # Eastlands
    "Eastleigh": (-1.2753, 36.8442),
    "Eastleigh Section 1": (-1.2708, 36.8472),
    "Eastleigh Section 2": (-1.2764, 36.8458),
    "Eastleigh Section 3": (-1.2792, 36.8444),
    "Garissa Lodge": (-1.2736, 36.8486),
    "First Avenue": (-1.2722, 36.8458),
    "General Waruinge Street": (-1.2750, 36.8472),
    
    # South Areas
    "South B": (-1.3142, 36.8297),
    "South C": (-1.3225, 36.8297),
    "Nyayo Stadium": (-1.3142, 36.8264),
    "Bellevue": (-1.3181, 36.8319),
    "Nairobi West": (-1.3264, 36.8208),
    "Madaraka": (-1.3097, 36.8264),
    "Nyayo Highrise": (-1.3125, 36.8278),
    
    # Industrial Area & Surroundings
    "Industrial Area": (-1.3208, 36.8472),
    "Enterprise Road": (-1.3236, 36.8486),
    "Likoni Road": (-1.3264, 36.8500),
    "Mombasa Road": (-1.3364, 36.8297),
    "Imara Daima": (-1.3542, 36.8583),
    "Nyayo Embakasi": (-1.3458, 36.8639),
    
    # Embakasi & Surroundings
    "Embakasi": (-1.3031, 36.8919),
    "Pipeline": (-1.3125, 36.8806),
    "Donholm": (-1.2958, 36.8944),
    "Umoja": (-1.2875, 36.8972),
    "Kariobangi": (-1.2653, 36.8806),
    "Komarock": (-1.2792, 36.9167),
    "Kayole": (-1.2736, 36.9194),
    "Mihango": (-1.3097, 36.8889),
    
    # Kasarani & Northern Areas
    "Kasarani": (-1.2258, 36.8969),
    "Mwiki": (-1.2125, 36.8944),
    "Githurai": (-1.1958, 36.9000),
    "Kahawa": (-1.1833, 36.9167),
    "Kahawa West": (-1.1806, 36.9139),
    "Zimmerman": (-1.2042, 36.8917),
    "Roysambu": (-1.2167, 36.8889),
    "Thome": (-1.2292, 36.8861),
    
    # Mathare & Surroundings
    "Mathare": (-1.2597, 36.8581),
    "Mathare North": (-1.2542, 36.8597),
    "Mathare Area 3": (-1.2625, 36.8611),
    "Huruma": (-1.2486, 36.8556),
    "Ngei": (-1.2458, 36.8583),
    "Kiamaiko": (-1.2514, 36.8639),
    
    # Kibera & Surroundings
    "Kibera": (-1.3133, 36.7919),
    "Olympic": (-1.3167, 36.7889),
    "Laini Saba": (-1.3181, 36.7861),
    "Makina": (-1.3208, 36.7833),
    "Soweto": (-1.3153, 36.7944),
    "Gatwekera": (-1.3125, 36.7972),
    
    # Major Roads & Highways
    "Ngong Road": (-1.3031, 36.7519),
    "Thika Road": (-1.2297, 36.8581),
    "Waiyaki Way": (-1.2676, 36.7672),
    "Jogoo Road": (-1.2875, 36.8472),
    "Outer Ring Road": (-1.2458, 36.8333),
    "Eastern Bypass": (-1.2792, 36.9000),
    "Southern Bypass": (-1.3542, 36.7806),
    "Northern Bypass": (-1.1958, 36.8500),
    
    # Universities & Institutions
    "University of Nairobi": (-1.2792, 36.8167),
    "Kenyatta University": (-1.1833, 36.9306),
    "USIU": (-1.2292, 36.8889),
    "Strathmore University": (-1.3097, 36.8153),
    "Daystar University": (-1.3458, 36.7361),
    
    # Hospitals
    "Kenyatta National Hospital": (-1.3014, 36.8069),
    "Nairobi Hospital": (-1.2931, 36.8097),
    "Aga Khan Hospital": (-1.2708, 36.8125),
    "MP Shah Hospital": (-1.2708, 36.8097),
    "Gertrude's Hospital": (-1.2653, 36.8069),
    
    # Shopping Centers & Malls
    "Village Market": (-1.2208, 36.8056),
    "Two Rivers Mall": (-1.2125, 36.8028),
    "Garden City Mall": (-1.2236, 36.8083),
    "Nextgen Mall": (-1.2208, 36.8111),
    "The Hub Karen": (-1.3236, 36.7069),
    "Prestige Plaza": (-1.2708, 36.8139),
    "T-Mall": (-1.2542, 36.8014),
    
    # Transport Hubs
    "Jomo Kenyatta International Airport": (-1.3192, 36.9278),
    "Wilson Airport": (-1.3208, 36.8153),
    "Railways Station": (-1.2847, 36.8278),
    "Country Bus Station": (-1.2806, 36.8194),
    "Machakos Bus Station": (-1.2819, 36.8222),
    "OTC Bus Station": (-1.2833, 36.8208),
    
    # Stages & Bus Stops
    "Kencom Stage": (-1.2847, 36.8194),
    "Koja Stage": (-1.2819, 36.8167),
    "Fire Station": (-1.2875, 36.8181),
    "GPO Stage": (-1.2847, 36.8208),
    "Railways Stage": (-1.2847, 36.8278),
    "Afya Centre": (-1.2875, 36.8194),
    "Muthurwa Market": (-1.2903, 36.8264),
    
    # Markets
    "City Market": (-1.2819, 36.8181),
    "Wakulima Market": (-1.2903, 36.8250),
    "Gikomba Market": (-1.2875, 36.8306),
    "Maasai Market": (-1.2847, 36.8194),
    "Muthurwa Market": (-1.2903, 36.8264),
    
    # Residential Areas
    "Runda": (-1.2042, 36.8028),
    "Muthaiga": (-1.2375, 36.8139),
    "Gigiri": (-1.2208, 36.8111),
    "Ridgeways": (-1.2125, 36.8167),
    "Loresho": (-1.2458, 36.7944),
    "Riverside": (-1.2708, 36.8000),
    "Milimani": (-1.2792, 36.8056),
    "Upper Hill": (-1.2931, 36.8125),
    "Kirichwa Road": (-1.2958, 36.8069),
    "State House Road": (-1.2875, 36.8056),
    
    # Satellite Towns
    "Ruiru": (-1.1458, 36.9583),
    "Kikuyu": (-1.2458, 36.6667),
    "Limuru": (-1.1167, 36.6417),
    "Kiambu": (-1.1714, 36.8356),
    "Thika": (-1.0333, 36.8667),
    "Machakos": (-1.5167, 37.2667),
    "Athi River": (-1.4500, 36.9833),
    
    # Other Notable Areas
    "Adams Arcade": (-1.3097, 36.7806),
    "Ngara": (-1.2625, 36.8306),
    "Pangani": (-1.2542, 36.8389),
    "Shauri Moyo": (-1.2708, 36.8389),
    "Ziwani": (-1.2736, 36.8417),
    "Kariokor": (-1.2708, 36.8333),
    "Bahati": (-1.2653, 36.8361),
    "California": (-1.2597, 36.8417),
    "Baba Dogo": (-1.2375, 36.8806),
    "Lucky Summer": (-1.2208, 36.8778),
    "Dandora": (-1.2458, 36.8972),
    "Kariobangi South": (-1.2681, 36.8833),
    "Kariobangi North": (-1.2625, 36.8861),
    "Buruburu": (-1.2875, 36.8750),
    "Jericho": (-1.2792, 36.8694),
    "Ofafa Jericho": (-1.2819, 36.8722),
    "Makadara": (-1.2958, 36.8583),
    "Harambee": (-1.2931, 36.8611),
    "Maringo": (-1.2903, 36.8556),
    "Kaloleni": (-1.2875, 36.8528),
    "Starehe": (-1.2792, 36.8361),
    "Pumwani": (-1.2736, 36.8361),
    "Eastleigh Airbase": (-1.2625, 36.8500),
    "Moi Air Base": (-1.2625, 36.8528)
}
# Initialize session state
if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = None
if 'user_history' not in st.session_state:
    st.session_state.user_history = []

# Load data and models at startup
df = load_data()
model, label_encoder = load_models()

if df is not None:
    df = preprocess_data(df)

# Title and header
#st.title("🚗 Nairobi Road Crash Risk Prediction & Hotspot Analysis Dashboard")
#st.markdown("---")

# Sidebar navigation
st.sidebar.header("Navigation")
page = st.sidebar.selectbox("Choose a page:", 
    ["Home", "Prediction", "Route Analysis", "Analytics", "Hotspots", "History", "About"])

from datetime import datetime

# Time-based personalized greeting
hour = datetime.now().hour
if hour < 12:
    greeting = "Good morning"
elif hour < 17:
    greeting = "Good afternoon"
else:
    greeting = "Good evening"

# HOME PAGE
if page == "Home":
    st.title("Nairobi Road Crash Risk & Hotspot Analysis")
    
    st.markdown(f"### {greeting}, welcome to your Road Safety Assistant")
    st.markdown("Make informed travel decisions using AI-powered crash risk predictions.")

    st.markdown("---")

    # Quick Start Guide
    st.subheader("Quick Start")
    st.markdown("""
    1. **Prediction** – Get crash risk for any location & time.  
    2. **Hotspots** – Explore Nairobi's high-risk crash zones.  
    3. **Analytics** – See patterns & trends in crash data.  
    """)

    st.markdown("---")

    # Brief Overview
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### Why This App?

        - Predict crash risk instantly.  
        - Check route safety before you leave.  
        - Discover Nairobi's crash hotspots.  
        - Use data to protect yourself & others.  
        """)
    with col2:
        if df is not None:
            total = len(df)
            severe = len(df[df['severity_category'] == 'Severe'])
            st.info(f"""
            **Data Analyzed**  
            {total:,} total crashes  
            {severe:,} severe cases  
            85%+ model accuracy
            """)

    st.markdown("---")

    # Features Summary
    st.subheader("Key Features")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Crash Prediction**\n\nEstimate crash risk by time & location.")
    with col2:
        st.markdown("**Hotspot Maps**\n\nFind crash-prone areas across Nairobi.")
    with col3:
        st.markdown("**Route Risk Check**\n\nPlan safer routes using risk insights.")

    st.markdown("---")

    # System Status
    st.subheader("System Stats")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Crashes", f"{len(df):,}" if df is not None else "N/A")
    with col2:
        if df is not None:
            sev = len(df[df['severity_category'] == 'Severe'])
            pct = (sev / len(df) * 100) if len(df) > 0 else 0
            st.metric("Severe", f"{sev:,}", f"{pct:.1f}%")
        else:
            st.metric("Severe", "N/A")
    with col3:
        st.metric("Model", "Ready" if model else "Error")
    with col4:
        st.metric("Your Runs", f"{len(st.session_state.user_history):,}")

    st.markdown("---")

    # Navigation Buttons
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Predict", type="primary"):
            st.session_state.page = "Prediction"
            st.rerun()
    with c2:
        if st.button("Hotspots"):
            st.session_state.page = "Hotspots"
            st.rerun()
    with c3:
        if st.button("Analytics"):
            st.session_state.page = "Analytics"
            st.rerun()


elif page == "Prediction":
    st.header("🎯 Advanced Crash Risk Prediction")
    
    if model is None or label_encoder is None:
        st.error("❌ Models not loaded. Please check your model files.")
    else:
        
        # Input form
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📅 Date & Time")
            selected_date = st.date_input("Date", datetime.now().date())
            selected_time = st.time_input("Time", time(12, 0))
            
        with col2:
            st.subheader("📍 Location Selection")
            
            location_method = st.radio(
                "Choose location method:",
                ["Quick Select", "Search by Name", "Manual Coordinates"]
            )
            
            latitude, longitude = -1.286389, 36.817222
            location_address = "Nairobi City Center"
            
            if location_method == "Quick Select":
                selected_location = st.selectbox(
                    "Select a known location:",
                    list(NAIROBI_LOCATIONS.keys())
                )
                latitude, longitude = NAIROBI_LOCATIONS[selected_location]
                location_address = selected_location
                
            elif location_method == "Search by Name":
                location_name = st.text_input(
                    "Enter road name, stage, or landmark:",
                    placeholder="e.g., Kenyatta Avenue, Westgate Mall, Ngong Road"
                )
                
                if location_name and st.button("🔍 Find Location"):
                    with st.spinner("Searching for location..."):
                        lat, lon, address = get_coordinates_from_location(location_name)
                        if lat and lon:
                            latitude, longitude = lat, lon
                            location_address = address
                            st.success(f"✅ Found: {address}")
                            st.session_state.found_location = {
                                'lat': lat, 'lon': lon, 'address': address
                            }
                        else:
                            st.error("❌ Location not found. Try a different name or use manual coordinates.")
                
                if 'found_location' in st.session_state and location_method == "Search by Name":
                    latitude = st.session_state.found_location['lat']
                    longitude = st.session_state.found_location['lon']
                    location_address = st.session_state.found_location['address']
                            
            else:
                latitude = st.number_input("Latitude", value=-1.286389, format="%.6f")
                longitude = st.number_input("Longitude", value=36.817222, format="%.6f")
        
        st.info(f"📍 Current location: {location_address}")
        st.write(f"Coordinates: {latitude:.6f}, {longitude:.6f}")
        
        # Prediction button
        if st.button("🔮 Predict Risk & Get Recommendations", type="primary"):
            with st.spinner("Analyzing crash risk..."):
                hour = selected_time.hour
                day_of_week = selected_date.weekday()
                month = selected_date.month
                rush_hour = 1 if (7 <= hour <= 9) or (16 <= hour <= 19) else 0
                weekend = 1 if day_of_week >= 5 else 0
                night_time = 1 if hour >= 22 or hour <= 5 else 0
                
                features = np.array([[hour, day_of_week, month, latitude, longitude, 
                                    rush_hour, weekend, night_time]])
                
                prediction = model.predict(features)[0]
                probabilities = model.predict_proba(features)[0]
                severity = label_encoder.inverse_transform([prediction])[0]
                confidence = max(probabilities)
                
                nearby_crashes = get_nearby_crashes(df, latitude, longitude, radius_km=2)
                
                traffic_factor = get_traffic_risk_factor(hour, day_of_week)
                risk_score = calculate_dynamic_risk_score(severity, confidence, nearby_crashes, traffic_factor=traffic_factor)
                risk_level, risk_color = get_risk_level(risk_score)
                
                recommendations = get_safety_recommendations(
                    severity, hour, weekend, rush_hour, night_time, nearby_crashes
                )
                
                st.session_state.prediction_results = {
                    'severity': severity,
                    'confidence': confidence,
                    'probabilities': probabilities,
                    'recommendations': recommendations,
                    'nearby_crashes': nearby_crashes,
                    'risk_score': risk_score,
                    'risk_level': risk_level,
                    'risk_color': risk_color,
                    'location': {
                        'lat': latitude,
                        'lon': longitude,
                        'address': location_address
                    },
                    'datetime': {
                        'date': selected_date,
                        'time': selected_time,
                        'hour': hour,
                        'weekend': weekend,
                        'rush_hour': rush_hour,
                        'night_time': night_time
                    }
                }
                
                save_to_history(st.session_state.prediction_results)
        
        # Display results
        if st.session_state.prediction_results is not None:
            results = st.session_state.prediction_results
            
            st.markdown("---")
            st.subheader("🎯 Prediction Results")
            
            # Main results display
            col_result1, col_result2, col_result3 = st.columns(3)
            
            with col_result1:
                severity = results['severity']
                if severity == 'Severe':
                    st.error(f"🚨 **Predicted Severity: {severity}**")
                elif severity == 'Moderate':
                    st.warning(f"⚠️ **Predicted Severity: {severity}**")
                else:
                    st.success(f"✅ **Predicted Severity: {severity}**")
            
            with col_result2:
                st.info(f"📊 **Confidence: {results['confidence']:.1%}**")
            
            with col_result3:
                risk_level = results['risk_level']
                if "EXTREME" in risk_level:
                    st.error(f"🔴 **{risk_level}**")
                elif "HIGH" in risk_level:
                    st.warning(f"🟠 **{risk_level}**")
                elif "MODERATE" in risk_level:
                    st.warning(f"🟡 **{risk_level}**")
                else:
                    st.success(f"🟢 **{risk_level}**")
            
            # Detailed analysis
            col_detail1, col_detail2 = st.columns(2)
            
            with col_detail1:
                # Risk score gauge
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = results['risk_score'],
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Dynamic Risk Score"},
                    delta = {'reference': 5},
                    gauge = {'axis': {'range': [None, 10]},
                             'bar': {'color': results['risk_color']},
                             'steps': [
                                 {'range': [0, 3], 'color': "lightgreen"},
                                 {'range': [3, 5], 'color': "yellow"},
                                 {'range': [5, 7], 'color': "orange"},
                                 {'range': [7, 10], 'color': "red"}],
                             'threshold': {'line': {'color': "black", 'width': 4},
                                          'thickness': 0.75, 'value': 8}}))
                st.plotly_chart(fig, use_container_width=True)
                
                # Risk score description
                st.markdown("""
                **How to Read the Dynamic Risk Score:**
                - **0-3**: 🟢 LOW RISK - Relatively safe conditions
                - **3-5**: 🟡 MODERATE RISK - Exercise normal caution  
                - **5-7**: 🟠 HIGH RISK - Increased vigilance required
                - **7-10**: 🔴 EXTREME RISK - Maximum caution needed
                
                The score considers predicted severity, model confidence, nearby crashes, and traffic conditions.
                """)
                
                # Probability distribution
                prob_df = pd.DataFrame({
                    'Severity': label_encoder.classes_,
                    'Probability': results['probabilities']
                })
                
                fig = px.bar(prob_df, x='Severity', y='Probability', 
                            title="Risk Probability Distribution",
                            color='Probability',
                            color_continuous_scale='RdYlGn_r')
                st.plotly_chart(fig, use_container_width=True)
            
            with col_detail2:
                # Safety recommendations
                st.subheader("🛡️ Safety Recommendations")
                for rec in results['recommendations']:
                    st.write(f"• {rec}")
                
                # Nearby crash statistics
                nearby_crashes = results['nearby_crashes']
                if len(nearby_crashes) > 0:
                    st.subheader("📊 Area Statistics (2km radius)")
                    st.write(f"• Total crashes nearby: {len(nearby_crashes)}")
                    
                    severity_counts = nearby_crashes['severity_category'].value_counts()
                    for sev, count in severity_counts.items():
                        st.write(f"• {sev} crashes: {count}")
                
                # Export report button
                st.subheader("📄 Export Report")
                report_text = generate_risk_report(results)
                st.download_button(
                    label="📥 Download Risk Assessment Report",
                    data=report_text,
                    file_name=f"risk_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
            
            # Interactive map
            st.subheader("🗺️ Location & Nearby Crash Analysis")
            
            location = results['location']
            m = folium.Map(location=[location['lat'], location['lon']], zoom_start=14)
            
            severity_colors = {
                'Severe': 'red',
                'Moderate': 'orange', 
                'Minor': 'yellow',
                'No_Injury': 'green'
            }
            
            # Prediction location
            folium.Marker(
                location=[location['lat'], location['lon']],
                popup=f"Prediction Location<br>Risk: {severity}<br>Score: {results['risk_score']:.1f}/10<br>Confidence: {results['confidence']:.1%}",
                icon=folium.Icon(color=severity_colors.get(severity, 'blue'), icon='star')
            ).add_to(m)
            
            # Nearby crashes
            if len(nearby_crashes) > 0:
                for _, crash in nearby_crashes.head(50).iterrows():
                    crash_color = severity_colors.get(crash['severity_category'], 'gray')
                    folium.CircleMarker(
                        location=[crash['latitude'], crash['longitude']],
                        radius=4,
                        color=crash_color,
                        fill=True,
                        popup=f"Past Crash<br>Severity: {crash['severity_category']}<br>Date: {crash['crash_date']}<br>Distance: {crash['distance_km']:.1f}km"
                    ).add_to(m)
            
            st_folium(m, width=700, height=400)
            
            if len(nearby_crashes) > 0:
                st.info(f"📍 Showing {min(50, len(nearby_crashes))} crashes within 2km of your location")
            else:
                st.info("📍 No recorded crashes within 2km of this location")
            
            # Clear results button
            if st.button("🔄 Clear Results & Make New Prediction"):
                st.session_state.prediction_results = None
                if 'found_location' in st.session_state:
                    del st.session_state.found_location
                st.rerun()

elif page == "Route Analysis":
    st.header("🛣️ Advanced Route Risk Analysis")
    
    if model is None or df is None:
        st.error("❌ Models or data not loaded.")
    else:
        st.write("Analyze crash risk along your planned route with temporal factors")
        
        # Date and Time Selection
        st.subheader("📅 Journey Details")
        col_dt1, col_dt2 = st.columns(2)
        
        with col_dt1:
            route_date = st.date_input("Journey Date", datetime.now().date(), key="route_date")
        with col_dt2:
            route_time = st.time_input("Journey Time", time(12, 0), key="route_time")
        
        # Location Selection
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🚀 Starting Location")
            start_method = st.radio("Choose start location method:", 
                                   ["Quick Select", "Search", "Use My Location", "Coordinates"], key="start")
            
            if start_method == "Quick Select":
                start_location = st.selectbox("Select starting location:", 
                                            list(NAIROBI_LOCATIONS.keys()), key="start_select")
                start_lat, start_lon = NAIROBI_LOCATIONS[start_location]
                start_address = start_location
                
            elif start_method == "Use My Location":
                st.info("📍 Click button to get your current location")
                if st.button("📍 Get My Current Location", key="get_start_location"):
                    st.info("🔄 Location detection would require browser permissions. Using default Nairobi center.")
                    start_lat, start_lon, start_address = -1.286389, 36.817222, "Nairobi City Center (Default)"
                else:
                    start_lat, start_lon, start_address = -1.286389, 36.817222, "Nairobi City Center (Default)"
                    
            elif start_method == "Search":
                start_name = st.text_input("Enter starting location:", key="start_search")
                if start_name and st.button("🔍 Find Start Location"):
                    with st.spinner("Searching..."):
                        lat, lon, address = get_coordinates_from_location(start_name)
                        if lat and lon:
                            start_lat, start_lon, start_address = lat, lon, address
                            st.success(f"✅ Found: {address}")
                        else:
                            st.error("❌ Location not found")
                            start_lat, start_lon, start_address = -1.286389, 36.817222, "Default"
                else:
                    start_lat, start_lon, start_address = -1.286389, 36.817222, "Default"
            else:
                start_lat = st.number_input("Start Latitude", value=-1.286389, format="%.6f", key="start_lat")
                start_lon = st.number_input("Start Longitude", value=36.817222, format="%.6f", key="start_lon")
                start_address = f"({start_lat:.4f}, {start_lon:.4f})"
        
        with col2:
            st.subheader("🎯 Destination")
            end_method = st.radio("Choose destination method:", 
                                 ["Quick Select", "Search", "Use My Location", "Coordinates"], key="end")
            
            if end_method == "Quick Select":
                end_location = st.selectbox("Select destination:", 
                                          list(NAIROBI_LOCATIONS.keys()), key="end_select")
                end_lat, end_lon = NAIROBI_LOCATIONS[end_location]
                end_address = end_location
                
            elif end_method == "Use My Location":
                st.info("📍 Click button to get your current location")
                if st.button("📍 Get My Current Location", key="get_end_location"):
                    st.info("🔄 Location detection would require browser permissions. Using default location.")
                    end_lat, end_lon, end_address = -1.3197, 36.7085, "Karen (Default)"
                else:
                    end_lat, end_lon, end_address = -1.3197, 36.7085, "Karen (Default)"
                    
            elif end_method == "Search":
                end_name = st.text_input("Enter destination:", key="end_search")
                if end_name and st.button("🔍 Find Destination"):
                    with st.spinner("Searching..."):
                        lat, lon, address = get_coordinates_from_location(end_name)
                        if lat and lon:
                            end_lat, end_lon, end_address = lat, lon, address
                            st.success(f"✅ Found: {address}")
                        else:
                            st.error("❌ Location not found")
                            end_lat, end_lon, end_address = -1.3197, 36.7085, "Default"
                else:
                    end_lat, end_lon, end_address = -1.3197, 36.7085, "Default"
            else:
                end_lat = st.number_input("End Latitude", value=-1.3197, format="%.6f", key="end_lat")
                end_lon = st.number_input("End Longitude", value=36.7085, format="%.6f", key="end_lon")
                end_address = f"({end_lat:.4f}, {end_lon:.4f})"
        
        # Route Summary
        st.info(f"🛣️ **Route:** {start_address} → {end_address}")
        st.info(f"📅 **Journey:** {route_date} at {route_time}")
        
        if st.button("🔍 Analyze Route Risk", type="primary"):
            with st.spinner("Analyzing route risk with temporal factors..."):
                # Calculate temporal factors
                hour = route_time.hour
                day_of_week = route_date.weekday()
                month = route_date.month
                rush_hour = 1 if (7 <= hour <= 9) or (16 <= hour <= 19) else 0
                weekend = 1 if day_of_week >= 5 else 0
                night_time = 1 if hour >= 22 or hour <= 5 else 0
                
                # Calculate route points and risks
                route_points, route_risks = calculate_route_risk(df, start_lat, start_lon, end_lat, end_lon)
                
                # Generate predictions for each route point
                route_predictions = []
                for i, ((lat, lon), historical_risk) in enumerate(zip(route_points, route_risks)):
                    features = np.array([[hour, day_of_week, month, lat, lon, rush_hour, weekend, night_time]])
                    prediction = model.predict(features)[0]
                    probabilities = model.predict_proba(features)[0]
                    severity = label_encoder.inverse_transform([prediction])[0]
                    confidence = max(probabilities)
                    
                    nearby_crashes = get_nearby_crashes(df, lat, lon, radius_km=1)
                    traffic_factor = get_traffic_risk_factor(hour, day_of_week)
                    risk_score = calculate_dynamic_risk_score(severity, confidence, nearby_crashes, traffic_factor=traffic_factor)
                    
                    route_predictions.append({
                        'point': i + 1,
                        'lat': lat,
                        'lon': lon,
                        'severity': severity,
                        'confidence': confidence,
                        'risk_score': risk_score,
                        'historical_risk': historical_risk,
                        'nearby_crashes': len(nearby_crashes)
                    })
                
                # Store route analysis in session state
                route_analysis = {
                    'timestamp': datetime.now(),
                    'start_address': start_address,
                    'end_address': end_address,
                    'journey_date': route_date,
                    'journey_time': route_time,
                    'route_points': route_predictions,
                    'temporal_factors': {
                        'hour': hour,
                        'rush_hour': rush_hour,
                        'weekend': weekend,
                        'night_time': night_time
                    }
                }
                
                # Initialize route history if not exists
                if 'route_history' not in st.session_state:
                    st.session_state.route_history = []
                
                st.session_state.route_history.append(route_analysis)
                st.session_state.current_route_analysis = route_analysis
                
                # Keep only last 20 route analyses
                if len(st.session_state.route_history) > 20:
                    st.session_state.route_history = st.session_state.route_history[-20:]
        
        # Display results if available
        if 'current_route_analysis' in st.session_state:
            analysis = st.session_state.current_route_analysis
            predictions = analysis['route_points']
            
            # Route statistics
            avg_risk = np.mean([p['risk_score'] for p in predictions])
            max_risk = np.max([p['risk_score'] for p in predictions])
            high_risk_segments = sum(1 for p in predictions if p['risk_score'] > 5)
            severe_predictions = sum(1 for p in predictions if p['severity'] == 'Severe')
            
            st.subheader("📊 Route Risk Analysis Results")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Average Risk Score", f"{avg_risk:.1f}/10")
            with col2:
                st.metric("Maximum Risk Point", f"{max_risk:.1f}/10")
            with col3:
                st.metric("High Risk Segments", high_risk_segments)
            with col4:
                st.metric("Severe Predictions", severe_predictions)
            
            # Temporal context
            factors = analysis['temporal_factors']
            context_info = []
            if factors['rush_hour']:
                context_info.append("🚦 Rush Hour")
            if factors['weekend']:
                context_info.append("📅 Weekend")
            if factors['night_time']:
                context_info.append("🌙 Night Time")
            
            if context_info:
                st.info(f"**Journey Context:** {' | '.join(context_info)}")
            
            # Route predictions table
            st.subheader("🎯 Point-by-Point Predictions")
            
            pred_df = pd.DataFrame(predictions)
            pred_df['risk_level'] = pred_df['risk_score'].apply(lambda x: get_risk_level(x)[0])
            
            # Style the dataframe
            def style_risk(val):
                if val >= 7:
                    return 'background-color: #ffebee'
                elif val >= 5:
                    return 'background-color: #fff3e0'
                elif val >= 3:
                    return 'background-color: #fffde7'
                else:
                    return 'background-color: #e8f5e8'
            
            styled_pred_df = pred_df[['point', 'severity', 'confidence', 'risk_score', 'risk_level', 'nearby_crashes']].style.applymap(style_risk, subset=['risk_score'])
            st.dataframe(styled_pred_df, use_container_width=True)
            
            # Route map with predictions
            st.subheader("🗺️ Interactive Route Risk Map")
            
            route_center_lat = (start_lat + end_lat) / 2
            route_center_lon = (start_lon + end_lon) / 2
            m = folium.Map(location=[route_center_lat, route_center_lon], zoom_start=12)
            
            # Add start and end markers
            folium.Marker([start_lat, start_lon], popup=f"🚀 Start: {start_address}", 
                        icon=folium.Icon(color='green', icon='play')).add_to(m)
            folium.Marker([end_lat, end_lon], popup=f"🎯 End: {end_address}", 
                        icon=folium.Icon(color='red', icon='stop')).add_to(m)
            
            # Add route points with risk-based colors
            for pred in predictions:
                risk_score = pred['risk_score']
                if risk_score >= 7:
                    color = 'red'
                elif risk_score >= 5:
                    color = 'orange'
                elif risk_score >= 3:
                    color = 'yellow'
                else:
                    color = 'green'
                
                folium.CircleMarker(
                    [pred['lat'], pred['lon']],
                    radius=8,
                    color=color,
                    fill=True,
                    popup=f"Point {pred['point']}<br>Severity: {pred['severity']}<br>Risk Score: {pred['risk_score']:.1f}<br>Confidence: {pred['confidence']:.1%}<br>Nearby Crashes: {pred['nearby_crashes']}"
                ).add_to(m)
            
            # Add route line
            route_coords = [[p['lat'], p['lon']] for p in predictions]
            folium.PolyLine(route_coords, color='blue', weight=3, opacity=0.7).add_to(m)
            
            st_folium(m, width=700, height=400)
            
            # Overall recommendations
            st.subheader("🛡️ Route Safety Recommendations")
            
            if avg_risk >= 6:
                st.error("🚨 **HIGH RISK ROUTE** - Consider alternative route or time")
                st.write("• Plan alternative route if possible")
                st.write("• Consider traveling at different time")
                st.write("• Exercise maximum caution")
            elif avg_risk >= 4:
                st.warning("⚠️ **MODERATE RISK ROUTE** - Exercise caution")
                st.write("• Drive defensively")
                st.write("• Maintain safe following distance")
                st.write("• Be extra alert in high-risk segments")
            else:
                st.success("✅ **RELATIVELY SAFE ROUTE**")
                st.write("• Maintain normal safety precautions")
                st.write("• Stay alert for changing conditions")
            
            if factors['rush_hour']:
                st.write("• 🚦 Rush hour traffic - expect delays and increased risk")
            if factors['night_time']:
                st.write("• 🌙 Night driving - ensure good visibility and reduce speed")
            if factors['weekend']:
                st.write("• 📅 Weekend travel - watch for recreational traffic patterns")
            
            # Export route analysis
            st.subheader("📄 Export Route Analysis")
            
            route_report = f"""NAIROBI ROUTE RISK ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

ROUTE DETAILS:
- From: {analysis['start_address']}
- To: {analysis['end_address']}
- Journey Date: {analysis['journey_date']}
- Journey Time: {analysis['journey_time']}

RISK SUMMARY:
- Average Risk Score: {avg_risk:.1f}/10
- Maximum Risk Point: {max_risk:.1f}/10
- High Risk Segments: {high_risk_segments}
- Severe Predictions: {severe_predictions}

TEMPORAL FACTORS:
- Rush Hour: {'Yes' if factors['rush_hour'] else 'No'}
- Weekend: {'Yes' if factors['weekend'] else 'No'}
- Night Time: {'Yes' if factors['night_time'] else 'No'}

POINT-BY-POINT ANALYSIS:
"""
            for pred in predictions:
                route_report += f"Point {pred['point']}: {pred['severity']} (Risk: {pred['risk_score']:.1f}, Confidence: {pred['confidence']:.1%})\n"
            
            route_report += "\nDISCLAIMER: This analysis is based on historical data and machine learning predictions. Always drive according to current conditions."
            
            st.download_button(
                label="📥 Download Route Analysis Report",
                data=route_report,
                file_name=f"route_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
            
            # Clear results button
            if st.button("🔄 Clear Results & Analyze New Route"):
                if 'current_route_analysis' in st.session_state:
                    del st.session_state.current_route_analysis
                st.rerun()

elif page == "Analytics":
    st.header("Advanced Crash Analytics Dashboard")
    
    if df is not None:
        # Time-based analysis
        st.subheader("Temporal Patterns")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Hourly distribution
            hourly_crashes = df.groupby('crash_hour').size()
            fig = px.bar(x=hourly_crashes.index, y=hourly_crashes.values,
                        title="Crashes by Hour of Day",
                        labels={'x': 'Hour', 'y': 'Number of Crashes'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Day of week distribution
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            daily_crashes = df.groupby('crash_dayofweek').size()
            fig = px.bar(x=[day_names[i] for i in daily_crashes.index], y=daily_crashes.values,
                        title="Crashes by Day of Week",
                        labels={'x': 'Day', 'y': 'Number of Crashes'})
            st.plotly_chart(fig, use_container_width=True)
        
        # Severity analysis
        st.subheader("Severity Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Severity distribution
            severity_counts = df['severity_category'].value_counts()
            fig = px.pie(values=severity_counts.values, names=severity_counts.index,
                        title="Crash Severity Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Severity by hour
            severity_hour = df.groupby(['crash_hour', 'severity_category']).size().unstack(fill_value=0)
            fig = px.bar(severity_hour, title="Severity Distribution by Hour",
                        labels={'index': 'Hour', 'value': 'Number of Crashes'})
            st.plotly_chart(fig, use_container_width=True)
        
        # Vehicle involvement
        st.subheader("Vehicle Involvement Analysis")
        
        vehicle_stats = {
            'Pedestrian': df['pedestrian_involved'].sum(),
            'Matatu': df['matatu_involved'].sum(),
            'Motorcycle': df['motorcycle_involved'].sum(),
            'Fatal': df['fatal_crash'].sum()
        }
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Pedestrian Involved", vehicle_stats['Pedestrian'])
        with col2:
            st.metric("Matatu Involved", vehicle_stats['Matatu'])
        with col3:
            st.metric("Motorcycle Involved", vehicle_stats['Motorcycle'])
        with col4:
            st.metric("Fatal Crashes", vehicle_stats['Fatal'])
        
        # Predictive insights
        st.subheader("Predictive Insights")
        
        col11, col12, col13 = st.columns(3)
        
        with col11:
            # Most dangerous hour
            dangerous_hour = df[df['severity_category'] == 'Severe']['crash_hour'].mode()[0]
            st.error(f"Most dangerous hour: **{dangerous_hour}:00**")
        
        with col12:
            # Weekend vs weekday
            weekend_severe = len(df[(df['weekend'] == 1) & (df['severity_category'] == 'Severe')])
            weekday_severe = len(df[(df['weekend'] == 0) & (df['severity_category'] == 'Severe')])
            if weekend_severe > weekday_severe:
                st.warning("**Weekends** more dangerous")
            else:
                st.info("**Weekdays** more dangerous")
        
        with col13:
            # Rush hour impact
            rush_severe = len(df[(df['rush_hour'] == 1) & (df['severity_category'] == 'Severe')])
            rush_risk = rush_severe / len(df[df['rush_hour'] == 1]) if len(df[df['rush_hour'] == 1]) > 0 else 0
            st.warning(f"Rush hour severe risk: **{rush_risk*100:.1f}%**")
    
    else:
        st.error("❌ No data available for analytics")

elif page == "Hotspots":
    st.header("🗺️ Crash Hotspot Analysis")
    
    if df is not None:
        # Simplified controls in a single row
        st.subheader("Detection Settings")
        col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
        
        with col1:
            eps_km = st.slider("Radius (km)", 0.1, 2.0, 0.5, 0.1)
        with col2:
            min_samples = st.slider("Min Crashes", 3, 20, 5)
        with col3:
            severity_filter = st.selectbox("Severity Filter", 
                                         ["All", "Severe", "Moderate", "Minor"])
        with col4:
            st.write("")  # spacing
            detect_btn = st.button("🔍 Detect", type="primary")
        
        # Filter data
        filtered_df = df.copy()
        if severity_filter != "All":
            filtered_df = df[df['severity_category'] == severity_filter]
        
        # Auto-detect or manual detect
        if detect_btn or 'hotspots_df' not in st.session_state:
            with st.spinner("Analyzing hotspots..."):
                hotspots_df = detect_crash_hotspots(filtered_df, eps_km, min_samples)
                st.session_state.hotspots_df = hotspots_df
        else:
            hotspots_df = st.session_state.hotspots_df
        
        if len(hotspots_df) > 0:
            st.success(f"✅ Found {len(hotspots_df)} hotspots")
            
            # Key metrics in cards
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("🎯 Total Hotspots", len(hotspots_df))
            with col2:
                high_risk = len(hotspots_df[hotspots_df['risk_score'] > 0.5])
                st.metric("🔴 High Risk", high_risk)
            with col3:
                total_crashes = hotspots_df['crash_count'].sum()
                st.metric("💥 Total Crashes", total_crashes)
            with col4:
                avg_risk = hotspots_df['risk_score'].mean()
                st.metric("📊 Avg Risk", f"{avg_risk:.2f}")
            
            # Interactive map first (most important)
            st.subheader("🗺️ Interactive Hotspot Map")
            
            valid_coords = filtered_df.dropna(subset=['latitude', 'longitude'])
            if len(valid_coords) > 0:
                center_lat = valid_coords['latitude'].mean()
                center_lon = valid_coords['longitude'].mean()
                
                m = folium.Map(location=[center_lat, center_lon], zoom_start=11)
                
                # Add hotspots with better styling
                for _, hotspot in hotspots_df.iterrows():
                    if hotspot['risk_score'] > 0.7:
                        color, icon = 'red', '⚠️'
                    elif hotspot['risk_score'] > 0.4:
                        color, icon = 'orange', '⚡'
                    else:
                        color, icon = 'yellow', '📍'
                    
                    folium.CircleMarker(
                        location=[hotspot['center_lat'], hotspot['center_lon']],
                        radius=min(hotspot['crash_count'] * 2, 20),
                        color=color,
                        fill=True,
                        popup=f"""
                        <b>{icon} Hotspot #{hotspot['cluster_id']+1}</b><br>
                        🚗 Crashes: {hotspot['crash_count']}<br>
                        ⚠️ Risk Score: {hotspot['risk_score']:.2f}<br>
                        🔴 Severe: {hotspot['severe_count']}
                        """
                    ).add_to(m)
                
                st_folium(m, width=700, height=500)
            
            # Top dangerous hotspots in a clean table
            st.subheader("Most Dangerous Hotspots")
            
            top_hotspots = hotspots_df.head(10).copy()
            
            # Create display dataframe
            display_df = pd.DataFrame({
                'Rank': range(1, len(top_hotspots) + 1),
                'Risk Level': top_hotspots['risk_score'].apply(lambda x: 
                    '🔴 EXTREME' if x > 0.7 else 
                    '🟠 HIGH' if x > 0.4 else 
                    '🟡 MODERATE'),
                'Total Crashes': top_hotspots['crash_count'],
                'Severe Crashes': top_hotspots['severe_count'],
                'Risk Score': top_hotspots['risk_score'].round(2),
                'Location': top_hotspots.apply(lambda row: 
                    f"{row['center_lat']:.4f}, {row['center_lon']:.4f}", axis=1)
            })
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            # Quick insights
            st.subheader("Key Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Analysis Summary:**")
                extreme_zones = len(hotspots_df[hotspots_df['risk_score'] > 0.7])
                if extreme_zones > 0:
                    st.error(f"{extreme_zones} EXTREME danger zones identified")
                
                high_zones = len(hotspots_df[hotspots_df['risk_score'] > 0.4])
                if high_zones > 0:
                    st.warning(f"{high_zones} HIGH risk areas need attention")
                
                coverage = (total_crashes / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
                st.info(f"{coverage:.1f}% of crashes occur in hotspots")
            
            with col2:
                st.markdown("**Recommendations:**")
                if extreme_zones > 0:
                    st.write("• 🚔 Deploy traffic police to red zones")
                    st.write("• 📹 Install speed cameras immediately")
                    st.write("• 🚧 Review road infrastructure")
                else:
                    st.write("• Continue current monitoring")
                    st.write("• Regular hotspot analysis")
                    st.write("• Update detection parameters")
        
        else:
            st.warning("No hotspots detected")
            st.info("Try adjusting the radius or minimum crashes settings")
    
    else:
        st.error("No data available for analysis")

elif page == "History":
    st.header("Prediction History")
    
    if st.session_state.user_history:
        history_df = pd.DataFrame(st.session_state.user_history)
        
        # Display history table
        st.subheader("Recent Predictions")
        
        # Style the dataframe
        def style_severity(val):
            if val == 'Severe':
                return 'background-color: #ffebee'
            elif val == 'Moderate':
                return 'background-color: #fff3e0'
            else:
                return 'background-color: #e8f5e8'
        
        styled_df = history_df.style.applymap(style_severity, subset=['severity'])
        
        st.dataframe(styled_df, use_container_width=True)
        
        # Trends analysis
        if len(history_df) > 1:
            st.subheader("Your Risk Trends")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Risk score over time
                fig = px.line(history_df, x='timestamp', y='risk_score',
                             title="Your Risk Scores Over Time",
                             labels={'timestamp': 'Date/Time', 'risk_score': 'Risk Score'})
                fig.add_hline(y=5, line_dash="dash", line_color="orange", 
                             annotation_text="High Risk Threshold")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Severity distribution
                severity_counts = history_df['severity'].value_counts()
                fig = px.pie(values=severity_counts.values, names=severity_counts.index,
                            title="Your Prediction Severity Distribution")
                st.plotly_chart(fig, use_container_width=True)
        
        # Export history
        st.subheader("Export History")
        
        if st.button("Download Prediction History"):
            csv = history_df.to_csv(index=False)
            st.download_button(
                label="Download as CSV",
                data=csv,
                file_name=f"prediction_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        # Clear history
        if st.button("Clear History", type="secondary"):
            if st.button("Confirm Clear History"):
                st.session_state.user_history = []
                st.success("History cleared!")
                st.rerun()
    
    else:
        st.info("No predictions made yet. Go to the Prediction page to start!")

elif page == "About":
    st.header("About Nairobi Crash Risk Predictor")

    st.markdown("""
    ### Overview

    The **Nairobi Crash Risk Predictor** is an AI-powered system designed to assess road crash risks across Nairobi using historical crash data. It supports safer travel by providing real-time risk scores and location-based insights.

    ### Model and Features

    - **Best Model:** XGBoost Classifier
    - **Risk Scoring:** Based on time, location, severity, and other variables
    - **Hotspot Analysis:** Using DBSCAN clustering
    - **Route Evaluation:** Highlights risk levels along selected routes
    - **Safety Tips:** Contextual recommendations to reduce crash exposure

    ### How It Works

    Crash reports are analyzed using geospatial and temporal features. The model identifies risk-prone areas and computes personalized risk assessments for drivers, helping make informed travel decisions.

    ### Technology Stack

    - **Modeling:** XGBoost, DBSCAN
    - **Data Tools:** pandas, numpy
    - **Geospatial:** Folium, GeoPy
    - **Interface:** Streamlit

    ### Who Can Benefit

    - **Drivers:** Plan routes with fewer risks
    - **Traffic Agencies:** Allocate patrols to risk zones
    - **City Planners:** Identify where safety infrastructure is needed

    ---
    **Note:** This system is built for educational purposes. Predictions are based on historical trends and should complement—not replace—caution and traffic rules.
    """)

    # System status
    st.subheader("System Status")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.success("Data Loaded" if df is not None else "Data Not Loaded")

    with col2:
        st.success("Models Loaded" if model is not None else "Models Not Loaded")

    with col3:
        st.info(f"{len(st.session_state.user_history)} Predictions Made")

    # Dataset metrics
    if df is not None:
        st.subheader("Dataset Statistics")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Records", f"{len(df):,}")

        with col2:
            date_range = (pd.to_datetime(df['crash_date'].max()) - pd.to_datetime(df['crash_date'].min())).days
            st.metric("Date Range", f"{date_range} days")

        with col3:
            valid_coords = len(df.dropna(subset=['latitude', 'longitude']))
            st.metric("Geo-located Crashes", f"{valid_coords:,}")

        with col4:
            severe_rate = len(df[df['severity_category'] == 'Severe']) / len(df) * 100
            st.metric("Severe Crash Rate", f"{severe_rate:.1f}%")

    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>Nairobi Crash Risk Predictor v2.0 | Built for Road Safety</p>
        <p>Last Updated: {datetime.now().strftime('%B %Y')}</p>
    </div>
    """, unsafe_allow_html=True)