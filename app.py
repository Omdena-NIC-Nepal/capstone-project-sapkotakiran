import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Set page configuration
st.set_page_config(page_title="Nepal Climate Analysis", page_icon="🌦️", layout="wide")

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #005CAF;
    }
    .info-text {
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">Climate Change Impact Assessment for Nepal</p>', unsafe_allow_html=True)
st.markdown('### Analyzing climate trends and predicting future patterns in Nepal')

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('nepal_climate_data_2000_2023.csv')
    
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    
    # Create date column
    df['date'] = pd.to_datetime({'year': df['YEAR'], 'month': df['MO'], 'day': df['DY']})
    
    # Create season column
    def get_season(month):
        if month in [12, 1, 2]:  # Winter
            return 'Winter'
        elif month in [3, 4, 5]:  # Spring
            return 'Spring'
        elif month in [6, 7, 8]:  # Summer/Monsoon
            return 'Summer'
        else:  # Fall
            return 'Fall'
    
    df['season'] = df['MO'].apply(get_season)
    
    # Create derived features
    df['temp_range'] = df['T2M_MAX'] - df['T2M_MIN']  # Daily temperature range
    
    # Define extreme weather thresholds
    extreme_heat_threshold = df['T2M_MAX'].quantile(0.95)
    extreme_cold_threshold = df['T2M_MIN'].quantile(0.05)
    heavy_rain_threshold = df['PRECTOTCORR'].quantile(0.95)
    
    # Mark extreme events
    df['extreme_heat'] = df['T2M_MAX'] > extreme_heat_threshold
    df['extreme_cold'] = df['T2M_MIN'] < extreme_cold_threshold
    df['heavy_rain'] = df['PRECTOTCORR'] > heavy_rain_threshold
    
    return df

# Load models
@st.cache_resource
def load_models():
    try:
        with open('temp_model.pkl', 'rb') as f:
            temp_model = pickle.load(f)
        with open('precip_model.pkl', 'rb') as f:
            precip_model = pickle.load(f)
        return temp_model, precip_model
    except FileNotFoundError:
        st.error("Model files not found. Please run the Jupyter notebook first to generate the models.")
        return None, None

# Load data and models
df = load_data()
temp_model, precip_model = load_models()

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a page:",
    ["Overview", "Temperature Analysis", "Precipitation Analysis", "Extreme Events", "Seasonal Patterns", "Future Predictions"]
)

# Overview page
if page == "Overview":
    st.markdown('<p class="sub-header">Overview of Nepal Climate Data (2000-2023)</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Dataset Information")
        st.write(f"**Time Period:** 2000-2023")
        st.write(f"**Number of Records:** {df.shape[0]:,}")
        st.write(f"**Locations:** Latitude {df['latitude'].min()} to {df['latitude'].max()}, Longitude {df['longitude'].min()} to {df['longitude'].max()}")
        
        # Display basic statistics
        st.markdown("### Key Climate Indicators")
        avg_temp = df['T2M'].mean()
        max_temp_ever = df['T2M_MAX'].max()
        min_temp_ever = df['T2M_MIN'].min()
        avg_precip = df['PRECTOTCORR'].mean()
        total_precip = df['PRECTOTCORR'].sum()
        
        st.write(f"**Average Temperature:** {avg_temp:.2f}°C")
        st.write(f"**Highest Recorded Temperature:** {max_temp_ever:.2f}°C")
        st.write(f"**Lowest Recorded Temperature:** {min_temp_ever:.2f}°C")
        st.write(f"**Average Daily Precipitation:** {avg_precip:.2f} mm")
        st.write(f"**Total Precipitation (2000-2023):** {total_precip:.2f} mm")
    
    with col2:
        st.markdown("### Data Sample")
        st.dataframe(df.head())
        
        # Map of Nepal with data points
        st.markdown("### Geographic Coverage")
        # Create a sample of data points for the map
        map_data = df.drop_duplicates(['latitude', 'longitude'])[['latitude', 'longitude']]
        st.map(map_data)

# Temperature Analysis page
elif page == "Temperature Analysis":
    st.markdown('<p class="sub-header">Temperature Trends Analysis</p>', unsafe_allow_html=True)
    
    # Annual temperature trends
    yearly_temp = df.groupby('YEAR')[['T2M', 'T2M_MAX', 'T2M_MIN']].mean().reset_index()
    
    st.markdown("### Annual Temperature Trends")
    fig = px.line(yearly_temp, x='YEAR', y=['T2M', 'T2M_MAX', 'T2M_MIN'],
                 labels={'value': 'Temperature (°C)', 'YEAR': 'Year', 'variable': 'Metric'},
                 title='Annual Temperature Trends in Nepal (2000-2023)',
                 color_discrete_map={'T2M': '#1E88E5', 'T2M_MAX': '#FF5722', 'T2M_MIN': '#4CAF50'})
    fig.update_layout(legend_title_text='Temperature Metric', 
                     legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True)
    
    # Temperature heatmap by month and year
    st.markdown("### Monthly Temperature Patterns")
    monthly_temp = df.groupby(['YEAR', 'MO'])['T2M'].mean().reset_index()
    monthly_temp_pivot = monthly_temp.pivot(index='YEAR', columns='MO', values='T2M')
    
    fig = px.imshow(monthly_temp_pivot,
                   labels=dict(x="Month", y="Year", color="Temperature (°C)"),
                   x=[f"Month {i}" for i in range(1, 13)],
                   y=monthly_temp_pivot.index,
                   aspect="auto",
                   color_continuous_scale="RdBu_r")
    fig.update_layout(title='Monthly Average Temperature Heatmap')
    st.plotly_chart(fig, use_container_width=True)
    
    # Temperature distribution
    st.markdown("### Temperature Distribution")
    temp_dist_options = st.selectbox(
        "Select Temperature Metric:",
        ["Average Temperature (T2M)", "Maximum Temperature (T2M_MAX)", "Minimum Temperature (T2M_MIN)", "Temperature Range"]
    )
    
    temp_col_map = {
        "Average Temperature (T2M)": "T2M",
        "Maximum Temperature (T2M_MAX)": "T2M_MAX",
        "Minimum Temperature (T2M_MIN)": "T2M_MIN",
        "Temperature Range": "temp_range"
    }
    
    selected_col = temp_col_map[temp_dist_options]
    
    fig = px.histogram(df, x=selected_col, nbins=50,
                      labels={selected_col: f"{temp_dist_options} (°C)"},
                      title=f"Distribution of {temp_dist_options} (2000-2023)")
    fig.update_layout(bargap=0.1)
    st.plotly_chart(fig, use_container_width=True)

# Precipitation Analysis page
elif page == "Precipitation Analysis":
    st.markdown('<p class="sub-header">Precipitation Patterns Analysis</p>', unsafe_allow_html=True)
    
    # Annual precipitation
    yearly_precip = df.groupby('YEAR')['PRECTOTCORR'].sum().reset_index()
    
    st.markdown("### Annual Precipitation Trends")
    fig = px.bar(yearly_precip, x='YEAR', y='PRECTOTCORR',
                labels={'PRECTOTCORR': 'Total Precipitation (mm)', 'YEAR': 'Year'},
                title='Annual Precipitation in Nepal (2000-2023)')
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
    
    # Monthly precipitation patterns
    st.markdown("### Monthly Precipitation Patterns")
    monthly_precip = df.groupby(['YEAR', 'MO'])['PRECTOTCORR'].sum().reset_index()
    monthly_precip_pivot = monthly_precip.pivot(index='YEAR', columns='MO', values='PRECTOTCORR')
    
    fig = px.imshow(monthly_precip_pivot,
                   labels=dict(x="Month", y="Year", color="Precipitation (mm)"),
                   x=[f"Month {i}" for i in range(1, 13)],
                   y=monthly_precip_pivot.index,
                   aspect="auto",
                   color_continuous_scale="Blues")
    fig.update_layout(title='Monthly Precipitation Heatmap')
    st.plotly_chart(fig, use_container_width=True)
    
    # Precipitation distribution
    st.markdown("### Precipitation Distribution")
    fig = px.histogram(df[df['PRECTOTCORR'] > 0], x='PRECTOTCORR', nbins=50,
                      labels={'PRECTOTCORR': 'Precipitation (mm)'},
                      title='Distribution of Daily Precipitation (2000-2023, non-zero values)')
    fig.update_layout(bargap=0.1)
    st.plotly_chart(fig, use_container_width=True)
    
    # Rainy days per year
    st.markdown("### Rainy Days Analysis")
    rainy_days = df.groupby('YEAR')['PRECTOTCORR'].apply(lambda x: (x > 0).sum()).reset_index()
    rainy_days.columns = ['YEAR', 'Rainy_Days']
    
    fig = px.line(rainy_days, x='YEAR', y='Rainy_Days',
                 labels={'Rainy_Days': 'Number of Rainy Days', 'YEAR': 'Year'},
                 title='Number of Rainy Days per Year (2000-2023)')
    fig.update_traces(mode='lines+markers')
    st.plotly_chart(fig, use_container_width=True)

# Extreme Events page
elif page == "Extreme Events":
    st.markdown('<p class="sub-header">Extreme Weather Events Analysis</p>', unsafe_allow_html=True)
    
    # Count extreme events by year
    extreme_events = df.groupby('YEAR').agg({
        'extreme_heat': 'sum',
        'extreme_cold': 'sum',
        'heavy_rain': 'sum'
    }).reset_index()
    
    st.markdown("### Extreme Weather Events Trends")
    fig = px.line(extreme_events, x='YEAR', y=['extreme_heat', 'extreme_cold', 'heavy_rain'],
                 labels={'value': 'Number of Days', 'YEAR': 'Year', 'variable': 'Event Type'},
                 title='Extreme Weather Events in Nepal (2000-2023)',
                 color_discrete_map={
                     'extreme_heat': '#FF5722',
                     'extreme_cold': '#2196F3',
                     'heavy_rain': '#673AB7'
                 })
    fig.update_layout(legend_title_text='Event Type', 
                     legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True)
    
    # Extreme events by season
    extreme_by_season = df.groupby('season').agg({
        'extreme_heat': 'sum',
        'extreme_cold': 'sum',
        'heavy_rain': 'sum'
    }).reset_index()
    
    st.markdown("### Extreme Events by Season")
    fig = px.bar(extreme_by_season, x='season', y=['extreme_heat', 'extreme_cold', 'heavy_rain'],
                barmode='group',
                labels={'value': 'Number of Days', 'season': 'Season', 'variable': 'Event Type'},
                title='Extreme Weather Events by Season (2000-2023)',
                color_discrete_map={
                    'extreme_heat': '#FF5722',
                    'extreme_cold': '#2196F3',
                    'heavy_rain': '#673AB7'
                })
    fig.update_layout(legend_title_text='Event Type', 
                     legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True)
    
    # Extreme event details
    st.markdown("### Extreme Event Details")
    event_type = st.selectbox(
        "Select Event Type:",
        ["Extreme Heat", "Extreme Cold", "Heavy Rain"]
    )
    
    event_col_map = {
        "Extreme Heat": "extreme_heat",
        "Extreme Cold": "extreme_cold",
        "Heavy Rain": "heavy_rain"
    }
    
    threshold_map = {
        "Extreme Heat": f"> {df['T2M_MAX'].quantile(0.95):.2f}°C",
        "Extreme Cold": f"< {df['T2M_MIN'].quantile(0.05):.2f}°C",
        "Heavy Rain": f"> {df['PRECTOTCORR'].quantile(0.95):.2f} mm"
    }
    
    selected_event = event_col_map[event_type]
    
    st.write(f"**{event_type} Threshold:** {threshold_map[event_type]}")
    st.write(f"**Total {event_type} Days (2000-2023):** {df[selected_event].sum()}")
    
    # Show events on calendar heatmap
    event_by_month_year = df.groupby(['YEAR', 'MO'])[selected_event].sum().reset_index()
    event_pivot = event_by_month_year.pivot(index='YEAR', columns='MO', values=selected_event)
    
    fig = px.imshow(event_pivot,
                   labels=dict(x="Month", y="Year", color=f"Days with {event_type}"),
                   x=[f"Month {i}" for i in range(1, 13)],
                   y=event_pivot.index,
                   aspect="auto",
                   color_continuous_scale="Reds")
    fig.update_layout(title=f'Monthly {event_type} Days Heatmap')
    st.plotly_chart(fig, use_container_width=True)

# Seasonal Patterns page
elif page == "Seasonal Patterns":
    st.markdown('<p class="sub-header">Seasonal Climate Patterns</p>', unsafe_allow_html=True)
    
    # Seasonal temperature trends
    seasonal_temp = df.groupby(['YEAR', 'season'])['T2M'].mean().reset_index()
    
    st.markdown("### Seasonal Temperature Trends")
    fig = px.line(seasonal_temp, x='YEAR', y='T2M', color='season',
                 labels={'T2M': 'Average Temperature (°C)', 'YEAR': 'Year'},
                 title='Seasonal Temperature Trends in Nepal (2000-2023)',
                 color_discrete_map={
                     'Winter': '#2196F3',
                     'Spring': '#4CAF50',
                     'Summer': '#FF5722',
                     'Fall': '#FF9800'
                 })
    fig.update_layout(legend_title_text='Season')
    st.plotly_chart(fig, use_container_width=True)
    
    # Seasonal precipitation trends
    seasonal_precip = df.groupby(['YEAR', 'season'])['PRECTOTCORR'].sum().reset_index()
    
    st.markdown("### Seasonal Precipitation Trends")
    fig = px.line(seasonal_precip, x='YEAR', y='PRECTOTCORR', color='season',
                 labels={'PRECTOTCORR': 'Total Precipitation (mm)', 'YEAR': 'Year'},
                 title='Seasonal Precipitation Trends in Nepal (2000-2023)',
                 color_discrete_map={
                     'Winter': '#2196F3',
                     'Spring': '#4CAF50',
                     'Summer': '#FF5722',
                     'Fall': '#FF9800'
                 })
    fig.update_layout(legend_title_text='Season')
    st.plotly_chart(fig, use_container_width=True)
    
    # Temperature distribution by season
    st.markdown("### Temperature Distribution by Season")
    fig = px.box(df, x='season', y='T2M',
                labels={'T2M': 'Temperature (°C)', 'season': 'Season'},
                title='Temperature Distribution by Season (2000-2023)',
                color='season',
                color_discrete_map={
                    'Winter': '#2196F3',
                    'Spring': '#4CAF50',
                    'Summer': '#FF5722',
                    'Fall': '#FF9800'
                })
    st.plotly_chart(fig, use_container_width=True)
    
    # Precipitation distribution by season
    st.markdown("### Precipitation Distribution by Season")
    fig = px.box(df[df['PRECTOTCORR'] > 0], x='season', y='PRECTOTCORR',
                labels={'PRECTOTCORR': 'Precipitation (mm)', 'season': 'Season'},
                title='Precipitation Distribution by Season (2000-2023, non-zero values)',
                color='season',
                color_discrete_map={
                    'Winter': '#2196F3',
                    'Spring': '#4CAF50',
                    'Summer': '#FF5722',
                    'Fall': '#FF9800'
                })
    fig.update_layout(yaxis_range=[0, df['PRECTOTCORR'].quantile(0.95) * 2])
    st.plotly_chart(fig, use_container_width=True)

# Future Predictions page
elif page == "Future Predictions":
    st.markdown('<p class="sub-header">Future Climate Predictions</p>', unsafe_allow_html=True)
    
    if temp_model is None or precip_model is None:
        st.error("Models not loaded. Please run the Jupyter notebook first to generate the models.")
    else:
        st.markdown("### Predict Future Climate")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # User inputs for prediction
            st.markdown("#### Select Prediction Parameters")
            pred_year = st.slider("Year", min_value=2024, max_value=2050, value=2025)
            pred_month = st.slider("Month", min_value=1, max_value=12, value=6)
            pred_day = st.slider("Day", min_value=1, max_value=31, value=15)
            pred_latitude = st.slider("Latitude", min_value=26.0, max_value=30.0, value=28.0, step=0.5)
            pred_longitude = st.slider("Longitude", min_value=80.0, max_value=88.0, value=84.0, step=0.5)
        
        with col2:
            # Create prediction data
            st.markdown("#### Prediction Results")
            
            try:
                pred_date = datetime(year=pred_year, month=pred_month, day=pred_day)
                day_of_year = pred_date.timetuple().tm_yday
                
                # Get season
                def get_season(month):
                    if month in [12, 1, 2]:  # Winter
                        return 'Winter'
                    elif month in [3, 4, 5]:  # Spring
                        return 'Spring'
                    elif month in [6, 7, 8]:  # Summer/Monsoon
                        return 'Summer'
                    else:  # Fall
                        return 'Fall'
                
                season = get_season(pred_month)
                
                # Create prediction dataframe
                pred_data = pd.DataFrame({
                    'year': [pred_year],
                    'month': [pred_month],
                    'day': [pred_day],
                    'day_of_year': [day_of_year],
                    'latitude': [pred_latitude],
                    'longitude': [pred_longitude],
                    'season_Fall': [1 if season == 'Fall' else 0],
                    'season_Spring': [1 if season == 'Spring' else 0],
                    'season_Summer': [1 if season == 'Summer' else 0],
                    'season_Winter': [1 if season == 'Winter' else 0]
                })
                
                # Make predictions
                temp_pred = temp_model.predict(pred_data)[0]
                precip_pred = precip_model.predict(pred_data)[0]
                
                # Display predictions
                st.metric("Predicted Temperature", f"{temp_pred:.2f}°C")
                st.metric("Predicted Precipitation", f"{precip_pred:.2f} mm")
                st.write(f"**Season:** {season}")
                st.write(f"**Location:** Lat {pred_latitude}, Long {pred_longitude}")
                
            except ValueError as e:
                st.error(f"Invalid date: {e}")
        
        # Future climate trends
        st.markdown("### Future Climate Trends (2024-2028)")
        
        # Create future prediction dataset using pandas date_range to avoid invalid dates
        future_dates = pd.date_range(start='2024-01-01', end='2028-12-31', freq='D')
        future_data = []

        for date in future_dates:
            year = date.year
            month = date.month
            day = date.day
            season = get_season(month)
            
            # Create a row for each location (simplified to one location for this example)
            future_data.append({
                'year': year,
                'month': month,
                'day': day,
                'day_of_year': date.dayofyear,
                'latitude': 28,  # Central Nepal
                'longitude': 84,
                'season_Fall': 1 if season == 'Fall' else 0,
                'season_Spring': 1 if season == 'Spring' else 0,
                'season_Summer': 1 if season == 'Summer' else 0,
                'season_Winter': 1 if season == 'Winter' else 0
            })

        future_df = pd.DataFrame(future_data)

        # Predict future temperatures and precipitation
        future_df['predicted_temp'] = temp_model.predict(future_df[['year', 'month', 'day', 'day_of_year', 'latitude', 'longitude',
                                                                'season_Fall', 'season_Spring', 'season_Summer', 'season_Winter']])
        future_df['predicted_precip'] = precip_model.predict(future_df[['year', 'month', 'day', 'day_of_year', 'latitude', 'longitude',
                                                                    'season_Fall', 'season_Spring', 'season_Summer', 'season_Winter']])

        # Aggregate by year and month for visualization
        future_monthly = future_df.groupby(['year', 'month']).agg({
            'predicted_temp': 'mean',
            'predicted_precip': 'sum'
        }).reset_index()

        # Create date column for plotting
        future_monthly['date'] = pd.to_datetime(future_monthly[['year', 'month']].assign(day=1))
        
        # Plot tabs for temperature and precipitation
        future_tabs = st.tabs(["Temperature Forecast", "Precipitation Forecast"])
        
        with future_tabs[0]:
            fig = px.line(future_monthly, x='date', y='predicted_temp',
                         labels={'predicted_temp': 'Temperature (°C)', 'date': 'Date'},
                         title='Predicted Monthly Average Temperature (2024-2028)')
            fig.update_traces(mode='lines+markers')
            st.plotly_chart(fig, use_container_width=True)
            
            # Annual temperature trend
            future_yearly = future_df.groupby('year')['predicted_temp'].mean().reset_index()
            fig = px.bar(future_yearly, x='year', y='predicted_temp',
                         labels={'predicted_temp': 'Temperature (°C)', 'year': 'Year'},
                         title='Predicted Annual Average Temperature (2024-2028)')
            st.plotly_chart(fig, use_container_width=True)
        
        with future_tabs[1]:
            fig = px.line(future_monthly, x='date', y='predicted_precip',
                         labels={'predicted_precip': 'Precipitation (mm)', 'date': 'Date'},
                         title='Predicted Monthly Total Precipitation (2024-2028)')
            fig.update_traces(mode='lines+markers')
            st.plotly_chart(fig, use_container_width=True)
            
            # Annual precipitation trend
            future_yearly = future_df.groupby('year')['predicted_precip'].sum().reset_index()
            fig = px.bar(future_yearly, x='year', y='predicted_precip',
                         labels={'predicted_precip': 'Precipitation (mm)', 'year': 'Year'},
                         title='Predicted Annual Total Precipitation (2024-2028)')
            st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<p style='text-align: center; color: gray;'>Climate Change Impact Assessment and Prediction System for Nepal | Data from 2000-2023</p>
""", unsafe_allow_html=True)