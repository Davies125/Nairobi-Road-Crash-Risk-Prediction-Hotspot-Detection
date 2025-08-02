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
    page_title="Nairobi Crash Risk Predictor",
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
    """Load the trained models"""
    try:
        model_names = [
            'best_model_random_forest.pkl',
            'model_random_forest.pkl',
            'best_model_xgboost.pkl',
            'model_xgboost.pkl'
        ]
        
        model = None
        for name in model_names:
            try:
                model = joblib.load(name)
                st.success(f"✅ Loaded model: {name}")
                break
            except FileNotFoundError:
                continue
        
        if model is None:
            st.error("❌ No model files found! Please add your trained model files.")
            return None, None
            
        label_encoder = joblib.load('label_encoder.pkl')
        st.success("✅ Loaded label encoder")
        
        return model, label_encoder
        
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
        return "🔴 EXTREME RISK", "red"
    elif risk_score >= 5:
        return "🟠 HIGH RISK", "orange"
    elif risk_score >= 3:
        return "🟡 MODERATE RISK", "yellow"
    else:
        return "🟢 LOW RISK", "green"

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
    
    lat_diff = (valid_coords['latitude'] - lat) * 111
    lon_diff = (valid_coords['longitude'] - lon) * 111 * np.cos(np.radians(lat))
    distance = np.sqrt(lat_diff**2 + lon_diff**2)
    
    nearby = valid_coords[distance <= radius_km].copy()
    nearby['distance_km'] = distance[distance <= radius_km]
    
    return nearby.sort_values('distance_km')

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
        recommendations.extend([
            "🚦 Rush hour traffic - expect congestion",
            "⏰ Allow extra travel time",
            "🚌 Watch for matatus making sudden stops"
        ])
    
    if is_night:
        recommendations.extend([
            "🌙 Night driving - use headlights and drive slower",
            "👁️ Extra caution for pedestrians (may be less visible)",
            "🔦 Ensure all lights are working properly"
        ])
    
    if is_weekend:
        recommendations.append("🍺 Weekend - watch for impaired drivers, especially at night")
    
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

# Common Nairobi locations for quick selection
NAIROBI_LOCATIONS = {
    "CBD/City Center": (-1.2864, 36.8172),
    "Westlands": (-1.2676, 36.8108),
    "Karen": (-1.3197, 36.7085),
    "Kilimani": (-1.2921, 36.7872),
    "Lavington": (-1.2836, 36.7672),
    "Kileleshwa": (-1.2836, 36.7672),
    "Parklands": (-1.2630, 36.8581),
    "Eastleigh": (-1.2753, 36.8442),
    "South B": (-1.3142, 36.8297),
    "South C": (-1.3225, 36.8297),
    "Langata": (-1.3515, 36.7519),
    "Kasarani": (-1.2258, 36.8969),
    "Embakasi": (-1.3031, 36.8919),
    "Kibera": (-1.3133, 36.7919),
    "Mathare": (-1.2597, 36.8581),
    "Ngong Road": (-1.3031, 36.7519),
    "Thika Road": (-1.2297, 36.8581),
    "Mombasa Road": (-1.3364, 36.8297),
    "Waiyaki Way": (-1.2676, 36.7672),
    "Uhuru Highway": (-1.2864, 36.8172)
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
    st.success(f"✅ Loaded {len(df):,} crash records")

# Title and header
st.title("🚗 Nairobi Road Crash Risk Prediction Dashboard")
st.markdown("### Advanced AI-Powered Road Safety Analysis System")
st.markdown("---")

# Sidebar navigation
st.sidebar.header("🧭 Navigation")
page = st.sidebar.selectbox("Choose a page:", 
    ["🏠 Home", "🎯 Prediction", "📊 Analytics", "🗺️ Hotspots", "📋 History", "ℹ️ About"])

if page == "🏠 Home":
    st.header("Welcome to Nairobi Crash Risk Predictor")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Crashes", f"{len(df):,}" if df is not None else "N/A")
    
    with col2:
        if df is not None:
            severe_crashes = len(df[df['severity_category'] == 'Severe'])
            st.metric("Severe Crashes", f"{severe_crashes:,}")
        else:
            st.metric("Severe Crashes", "N/A")
    
    with col3:
        st.metric("Models Loaded", "✅" if model is not None else "❌")
    
    with col4:
        st.metric("Predictions Made", len(st.session_state.user_history))
    
    st.markdown("""
    ### 🎯 What This App Does:
    - **Predicts crash risk** using advanced machine learning models
    - **Identifies hotspots** using clustering algorithms
    - **Provides safety recommendations** based on real-time analysis
    - **Analyzes crash patterns** with interactive visualizations
    - **Generates detailed reports** for risk assessment
    
    ### 📋 How to Use:
    1. **Prediction**: Get instant risk assessments for any location and time
    2. **Analytics**: Explore crash patterns and trends
    3. **Hotspots**: Discover dangerous areas with advanced clustering
    4. **History**: Review your previous predictions
    5. **Reports**: Download detailed risk assessment reports
    
    ### 🚀 New Features:
    - Dynamic risk scoring with multiple factors
    - Advanced hotspot detection using DBSCAN clustering
    - Enhanced analytics with heatmaps and trends
    - User history and report generation
    - Improved mobile-responsive design
    """)
    
    if df is not None:
        # Quick stats
        st.subheader("📊 Quick Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            peak_hour = df['crash_hour'].mode()[0]
            st.info(f"🕐 Peak crash hour: **{peak_hour}:00**")
        
        with col2:
            weekend_crashes = len(df[df['weekend'] == 1])
            weekday_crashes = len(df[df['weekend'] == 0])
            if weekend_crashes > weekday_crashes:
                st.warning("📅 **Weekends** are more dangerous")
            else:
                st.info("📅 **Weekdays** have more crashes")
        
        with col3:
            severe_rate = len(df[df['severity_category'] == 'Severe']) / len(df) * 100
            st.error(f"🚨 Severe crash rate: **{severe_rate:.1f}%**")

elif page == "🎯 Prediction":
    st.header("🎯 Advanced Crash Risk Prediction")
    
    if model is None or label_encoder is None:
        st.error("❌ Models not loaded. Please check your model files.")
    else:
        st.success("✅ Ready for predictions!")
        
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
                if results['risk_color'] == 'red':
                    st.error(f"**{results['risk_level']}**")
                elif results['risk_color'] == 'orange':
                    st.warning(f"**{results['risk_level']}**")
                elif results['risk_color'] == 'yellow':
                    st.warning(f"**{results['risk_level']}**")
                else:
                    st.success(f"**{results['risk_level']}**")
            
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

elif page == "📊 Analytics":
    st.header("📊 Advanced Crash Analytics Dashboard")
    
    if df is not None:
        # Time-based analysis
        st.subheader("⏰ Temporal Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Hourly heatmap
            hourly_daily = df.pivot_table(
                values='crash_datetime', 
                index='crash_dayofweek', 
                columns='crash_hour', 
                aggfunc='count', 
                fill_value=0
            )
            
            day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            fig = px.imshow(
                hourly_daily.values,
                x=list(range(24)),
                y=day_names,
                title="Crash Frequency Heatmap (Hour vs Day)",
                color_continuous_scale='Reds',
                labels={'x': 'Hour of Day', 'y': 'Day of Week', 'color': 'Crashes'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Severity trends over time
            monthly_severity = df.groupby(['crash_month', 'severity_category']).size().unstack(fill_value=0)
            fig = px.line(
                monthly_severity,
                title="Monthly Severity Trends",
                labels={'index': 'Month', 'value': 'Number of Crashes'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Geographic analysis
        st.subheader("🗺️ Geographic Risk Analysis")
        
        col3, col4 = st.columns(2)
        
        with col3:
            # Hourly crashes
            hourly = df['crash_hour'].value_counts().sort_index()
            fig = px.line(x=hourly.index, y=hourly.values, 
                         title="Crashes by Hour of Day",
                         labels={'x': 'Hour', 'y': 'Number of Crashes'})
            fig.add_vline(x=8, line_dash="dash", line_color="red", annotation_text="Morning Rush")
            fig.add_vline(x=17, line_dash="dash", line_color="red", annotation_text="Evening Rush")
            st.plotly_chart(fig, use_container_width=True)
        
        with col4:
            # Severity distribution
            severity_counts = df['severity_category'].value_counts()
            fig = px.pie(values=severity_counts.values, names=severity_counts.index,
                        title="Crash Severity Distribution",
                        color_discrete_map={
                            'Severe': 'red',
                            'Moderate': 'orange',
                            'Minor': 'yellow',
                            'No_Injury': 'green'
                        })
            st.plotly_chart(fig, use_container_width=True)
        
        # Additional analytics
        col5, col6 = st.columns(2)
        
        with col5:
            # Day of week distribution
            day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            daily_crashes = df['crash_dayofweek'].value_counts().sort_index()
            fig = px.bar(x=[day_names[i] for i in daily_crashes.index], 
                        y=daily_crashes.values,
                        title="Crashes by Day of Week",
                        labels={'x': 'Day of Week', 'y': 'Number of Crashes'},
                        color=daily_crashes.values,
                        color_continuous_scale='Reds')
            st.plotly_chart(fig, use_container_width=True)
        
        with col6:
            # Monthly distribution
            monthly_crashes = df['crash_month'].value_counts().sort_index()
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            fig = px.bar(x=[month_names[i-1] for i in monthly_crashes.index], 
                        y=monthly_crashes.values,
                        title="Crashes by Month",
                        labels={'x': 'Month', 'y': 'Number of Crashes'},
                        color=monthly_crashes.values,
                        color_continuous_scale='Reds')
            st.plotly_chart(fig, use_container_width=True)
        
        # Key statistics
        st.subheader("📈 Key Insights")
        col7, col8, col9, col10 = st.columns(4)
        
        with col7:
            peak_hour = df['crash_hour'].mode()[0]
            st.metric("Peak Crash Hour", f"{peak_hour}:00")
        
        with col8:
            rush_hour_crashes = len(df[df['rush_hour'] == 1])
            rush_hour_rate = rush_hour_crashes / len(df) * 100
            st.metric("Rush Hour Crashes", f"{rush_hour_crashes:,}", f"{rush_hour_rate:.1f}%")
        
        with col9:
            weekend_crashes = len(df[df['weekend'] == 1])
            weekend_rate = weekend_crashes / len(df) * 100
            st.metric("Weekend Crashes", f"{weekend_crashes:,}", f"{weekend_rate:.1f}%")
        
        with col10:
            night_crashes = len(df[df['night_time'] == 1])
            night_rate = night_crashes / len(df) * 100
            st.metric("Night Time Crashes", f"{night_crashes:,}", f"{night_rate:.1f}%")
        
        # Predictive insights
        st.subheader("🔮 Predictive Insights")
        
        col11, col12, col13 = st.columns(3)
        
        with col11:
            # Most dangerous hour
            dangerous_hour = df[df['severity_category'] == 'Severe']['crash_hour'].mode()[0]
            st.error(f"🕐 Most dangerous hour: **{dangerous_hour}:00**")
        
        with col12:
            # Weekend vs weekday risk
            weekend_severe = len(df[(df['weekend'] == 1) & (df['severity_category'] == 'Severe')])
            weekday_severe = len(df[(df['weekend'] == 0) & (df['severity_category'] == 'Severe')])
            weekend_risk = weekend_severe / len(df[df['weekend'] == 1]) if len(df[df['weekend'] == 1]) > 0 else 0
            weekday_risk = weekday_severe / len(df[df['weekend'] == 0]) if len(df[df['weekend'] == 0]) > 0 else 0
            
            if weekend_risk > weekday_risk:
                st.warning(f"📅 Weekends are **{((weekend_risk/weekday_risk-1)*100):.1f}%** more dangerous")
            else:
                st.info(f"📅 Weekdays have **{((weekday_risk/weekend_risk-1)*100):.1f}%** higher severe risk")
        
        with col13:
            # Rush hour impact
            rush_severe = len(df[(df['rush_hour'] == 1) & (df['severity_category'] == 'Severe')])
            rush_risk = rush_severe / len(df[df['rush_hour'] == 1]) if len(df[df['rush_hour'] == 1]) > 0 else 0
            st.warning(f"🚦 Rush hour severe risk: **{rush_risk*100:.1f}%**")
    
    else:
        st.error("❌ No data available for analytics")

elif page == "🗺️ Hotspots":
    st.header("🗺️ Advanced Crash Hotspot Analysis")
    
    if df is not None:
        # Hotspot detection controls
        st.subheader("🔧 Hotspot Detection Parameters")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            eps_km = st.slider("Hotspot Radius (km)", 0.1, 2.0, 0.5, 0.1,
                              help="Radius for grouping nearby crashes into hotspots")
        with col2:
            min_samples = st.slider("Min Crashes per Hotspot", 3, 20, 5,
                                   help="Minimum number of crashes to form a hotspot")
        with col3:
            severity_filter = st.selectbox("Filter by Severity", 
                                         ["All", "Severe", "Moderate", "Minor"])
        
        # Filter data if needed
        filtered_df = df.copy()
        if severity_filter != "All":
            filtered_df = df[df['severity_category'] == severity_filter]
        
        # Detect hotspots
        with st.spinner("Detecting crash hotspots..."):
            hotspots_df = detect_crash_hotspots(filtered_df, eps_km, min_samples)
        
        if len(hotspots_df) > 0:
            # Display hotspot statistics
            st.subheader("📊 Hotspot Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Hotspots", len(hotspots_df))
            with col2:
                high_risk_hotspots = len(hotspots_df[hotspots_df['risk_score'] > 0.5])
                st.metric("High Risk Hotspots", high_risk_hotspots)
            with col3:
                total_crashes_in_hotspots = hotspots_df['crash_count'].sum()
                st.metric("Crashes in Hotspots", total_crashes_in_hotspots)
            with col4:
                avg_risk = hotspots_df['risk_score'].mean()
                st.metric("Average Risk Score", f"{avg_risk:.2f}")
            
            # Top 10 most dangerous hotspots
            st.subheader("🚨 Top 10 Most Dangerous Hotspots")
            top_hotspots = hotspots_df.head(10)
            
            for idx, hotspot in top_hotspots.iterrows():
                risk_color = "🔴" if hotspot['risk_score'] > 0.7 else "🟠" if hotspot['risk_score'] > 0.4 else "🟡"
                
                with st.expander(f"{risk_color} Hotspot #{hotspot['cluster_id']+1} - Risk Score: {hotspot['risk_score']:.2f}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"📍 **Location:** {hotspot['center_lat']:.6f}, {hotspot['center_lon']:.6f}")
                        st.write(f"💥 **Total Crashes:** {hotspot['crash_count']}")
                        st.write(f"🚨 **Severe Crashes:** {hotspot['severe_count']}")
                    with col2:
                        st.write(f"⚠️ **Risk Score:** {hotspot['risk_score']:.2f}")
                        risk_percentage = hotspot['risk_score'] * 100
                        st.write(f"📊 **Severity Rate:** {risk_percentage:.1f}%")
                        
                        # Risk level
                        if hotspot['risk_score'] > 0.7:
                            st.error("🔴 EXTREME DANGER ZONE")
                        elif hotspot['risk_score'] > 0.4:
                            st.warning("🟠 HIGH RISK AREA")
                        else:
                            st.info("🟡 MODERATE RISK AREA")
            
            # Interactive map with hotspots
            st.subheader("🗺️ Interactive Hotspot Map")
            
            valid_coords = filtered_df.dropna(subset=['latitude', 'longitude'])
            if len(valid_coords) > 0:
                center_lat = valid_coords['latitude'].mean()
                center_lon = valid_coords['longitude'].mean()
                
                m = folium.Map(location=[center_lat, center_lon], zoom_start=11)
                
                # Add hotspot circles
                for _, hotspot in hotspots_df.iterrows():
                    # Color based on risk score
                    if hotspot['risk_score'] > 0.7:
                        color = 'red'
                        fillColor = 'red'
                    elif hotspot['risk_score'] > 0.4:
                        color = 'orange'
                        fillColor = 'orange'
                    else:
                        color = 'yellow'
                        fillColor = 'yellow'
                    
                    folium.Circle(
                        location=[hotspot['center_lat'], hotspot['center_lon']],
                        radius=eps_km * 1000,
                        color=color,
                        fill=True,
                        fillColor=fillColor,
                        fillOpacity=0.3,
                        popup=f"<b>Hotspot #{hotspot['cluster_id']+1}</b><br>"
                              f"Crashes: {hotspot['crash_count']}<br>"
                              f"Severe: {hotspot['severe_count']}<br>"
                              f"Risk Score: {hotspot['risk_score']:.2f}<br>"
                              f"Severity Rate: {hotspot['risk_score']*100:.1f}%",
                        tooltip=f"Hotspot #{hotspot['cluster_id']+1} (Risk: {hotspot['risk_score']:.2f})"
                    ).add_to(m)
                    
                    # Add center marker
                    folium.Marker(
                        location=[hotspot['center_lat'], hotspot['center_lon']],
                        icon=folium.Icon(color='black', icon='exclamation-triangle'),
                        popup=f"Hotspot Center #{hotspot['cluster_id']+1}"
                    ).add_to(m)
                
                # Add individual crashes
                for _, crash in valid_coords.head(200).iterrows():
                    severity_colors = {'Severe': 'red', 'Moderate': 'orange', 'Minor': 'yellow', 'No_Injury': 'green'}
                    crash_color = severity_colors.get(crash['severity_category'], 'gray')
                    
                    folium.CircleMarker(
                        location=[crash['latitude'], crash['longitude']],
                        radius=2,
                        color=crash_color,
                        fill=True,
                        popup=f"Crash: {crash['severity_category']}<br>Date: {crash['crash_date']}"
                    ).add_to(m)
                
                # Add legend
                legend_html = '''
                <div style="position: fixed; 
                            bottom: 50px; left: 50px; width: 200px; height: 160px; 
                            background-color: white; border:2px solid grey; z-index:9999; 
                            font-size:12px; padding: 10px">
                <p><b>Hotspot Legend</b></p>
                <p><i class="fa fa-circle" style="color:red"></i> Extreme Risk (>70%)</p>
                <p><i class="fa fa-circle" style="color:orange"></i> High Risk (40-70%)</p>
                <p><i class="fa fa-circle" style="color:yellow"></i> Moderate Risk (<40%)</p>
                <p><i class="fa fa-exclamation-triangle" style="color:black"></i> Hotspot Center</p>
                <p><i class="fa fa-circle" style="color:red"></i> Individual Crashes</p>
                </div>
                '''
                m.get_root().html.add_child(folium.Element(legend_html))
                
                st_folium(m, width=700, height=500)
                
                st.info(f"📍 Showing {len(hotspots_df)} hotspots and {min(200, len(valid_coords))} individual crashes")
            
            # Hotspot analysis summary
            st.subheader("📋 Hotspot Analysis Summary")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**🎯 Key Findings:**")
                st.write(f"• {len(hotspots_df)} crash hotspots identified")
                st.write(f"• {high_risk_hotspots} high-risk zones require immediate attention")
                st.write(f"• {total_crashes_in_hotspots} crashes occurred in hotspot areas")
                st.write(f"• Average risk score: {avg_risk:.2f}")
            
            with col2:
                st.write("**🛡️ Recommendations:**")
                if high_risk_hotspots > 0:
                    st.write("• Deploy additional traffic enforcement in red zones")
                    st.write("• Install speed cameras and warning signs")
                    st.write("• Improve road infrastructure in hotspot areas")
                    st.write("• Conduct safety awareness campaigns")
                else:
                    st.write("• Continue monitoring identified hotspots")
                    st.write("• Maintain current safety measures")
                    st.write("• Regular review of hotspot parameters")
            
        else:
            st.warning("⚠️ No hotspots detected with current parameters.")
            st.info("💡 Try adjusting the radius or minimum samples, or check if there's sufficient data.")
    
    else:
        st.error("❌ No data available for hotspot analysis")

elif page == "📋 History":
    st.header("📋 Prediction History")
    
    if len(st.session_state.user_history) > 0:
        st.subheader(f"📊 Your Last {len(st.session_state.user_history)} Predictions")
        
        # Convert history to DataFrame for better display
        history_df = pd.DataFrame(st.session_state.user_history)
        history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
        history_df = history_df.sort_values('timestamp', ascending=False)
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Predictions", len(history_df))
        
        with col2:
            avg_confidence = history_df['confidence'].mean()
            st.metric("Avg Confidence", f"{avg_confidence:.1%}")
        
        with col3:
            severe_predictions = len(history_df[history_df['severity'] == 'Severe'])
            st.metric("Severe Risk Predictions", severe_predictions)
        
        with col4:
            avg_risk_score = history_df['risk_score'].mean()
            st.metric("Avg Risk Score", f"{avg_risk_score:.1f}/10")
        
        # History table
        st.subheader("📝 Detailed History")
        
        # Format the display
        display_df = history_df.copy()
        display_df['Time'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
        display_df['Confidence'] = display_df['confidence'].apply(lambda x: f"{x:.1%}")
        display_df['Risk Score'] = display_df['risk_score'].apply(lambda x: f"{x:.1f}/10")
        
        # Color code severity
        def color_severity(val):
            if val == 'Severe':
                return 'background-color: #ffebee'
            elif val == 'Moderate':
                return 'background-color: #fff3e0'
            elif val == 'Minor':
                return 'background-color: #f9fbe7'
            else:
                return 'background-color: #e8f5e8'
        
        styled_df = display_df[['Time', 'location', 'severity', 'Confidence', 'Risk Score']].style.applymap(
            color_severity, subset=['severity']
        )
        
        st.dataframe(styled_df, use_container_width=True)
        
        # Trends analysis
        if len(history_df) > 1:
            st.subheader("📈 Your Risk Trends")
            
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
        st.subheader("📥 Export History")
        
        if st.button("📄 Download Prediction History"):
            csv = history_df.to_csv(index=False)
            st.download_button(
                label="💾 Download as CSV",
                data=csv,
                file_name=f"prediction_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        # Clear history
        if st.button("🗑️ Clear History", type="secondary"):
            if st.button("⚠️ Confirm Clear History"):
                st.session_state.user_history = []
                st.success("✅ History cleared!")
                st.rerun()
    
    else:
        st.info("📭 No predictions made yet. Go to the Prediction page to start!")
        
        if st.button("🎯 Go to Prediction Page"):
            st.switch_page("🎯 Prediction")

elif page == "ℹ️ About":
    st.header("ℹ️ About Nairobi Crash Risk Predictor")
    
    st.markdown("""
    ### 🎯 Project Overview
    
    The **Nairobi Crash Risk Predictor** is an advanced AI-powered system designed to predict and analyze road crash risks in Nairobi, Kenya. Using machine learning algorithms and historical crash data, it provides real-time risk assessments and safety recommendations.
    
    ### 🔬 Technical Details
    
    **Machine Learning Models:**
    - Random Forest Classifier
    - XGBoost Classifier
    - Feature engineering with temporal and spatial variables
    - Dynamic risk scoring with multiple factors
    
    **Key Features:**
    - Real-time crash risk prediction
    - Advanced hotspot detection using DBSCAN clustering
    - Interactive visualizations and maps
    - Safety recommendations based on context
    - Historical analysis and trends
    - Downloadable risk assessment reports
    
    ### 📊 Data Sources
    
    - Historical crash data from Nairobi traffic authorities
    - Geographic coordinates and location information
    - Temporal patterns (hour, day, month)
    - Crash severity classifications
    - Vehicle type involvement data
    
    ### 🛠️ Technology Stack
    
    - **Frontend:** Streamlit
    - **Machine Learning:** scikit-learn, XGBoost
    - **Data Processing:** pandas, numpy
    - **Visualization:** Plotly, Folium
    - **Geospatial:** GeoPy, Folium
    - **Clustering:** DBSCAN algorithm
    
    ### 📈 Model Performance
    
    Our models have been trained and validated on historical crash data with the following considerations:
    - Cross-validation for model reliability
    - Feature importance analysis
    - Temporal validation to ensure robustness
    - Regular model updates with new data
    
    ### 🎯 Use Cases
    
    **For Drivers:**
    - Get risk assessments before traveling
    - Receive safety recommendations
    - Plan safer routes and timing
    
    **For Traffic Authorities:**
    - Identify high-risk areas for intervention
    - Deploy resources more effectively
    - Monitor crash patterns and trends
    
    **For Urban Planners:**
    - Understand crash hotspots for infrastructure planning
    - Analyze temporal patterns for traffic management
    - Support evidence-based decision making
    
    ### ⚠️ Important Disclaimers
    
    - This system provides risk assessments based on historical data and should not be the sole factor in safety decisions
    - Always follow traffic rules and drive according to current road conditions
    - Risk predictions are probabilistic and cannot guarantee safety outcomes
    - The system is designed to supplement, not replace, human judgment and official traffic guidance
    
    ### 🔮 Future Enhancements
    
    - Real-time weather integration
    - Live traffic data incorporation
    - Mobile app development
    - Integration with navigation systems
    - Expanded coverage to other Kenyan cities
    - Community reporting features
    
    ### 👥 Development Team
    
    This project was developed as part of advanced data science and machine learning research focused on improving road safety in urban environments.
    
    ### 📞 Contact & Support
    
    For technical support, feature requests, or data inquiries, please contact the development team.
    
    ---
    
    **Version:** 2.0  
    **Last Updated:** {datetime.now().strftime('%B %Y')}  
    **License:** Educational and Research Use
    """)
    
    # System status
    st.subheader("🔧 System Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.success("✅ Data Loaded" if df is not None else "❌ Data Not Loaded")
    
    with col2:
        st.success("✅ Models Loaded" if model is not None else "❌ Models Not Loaded")
    
    with col3:
        st.info(f"📊 {len(st.session_state.user_history)} Predictions Made")
    
    # Performance metrics (if available)
    if df is not None:
        st.subheader("📊 Dataset Statistics")
        
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
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>🚗 Nairobi Crash Risk Predictor v2.0 | Built with ❤️ for Road Safety</p>
    <p>⚠️ Always drive safely and follow traffic rules</p>
</div>
""", unsafe_allow_html=True)