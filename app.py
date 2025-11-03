import streamlit as st
import pandas as pd
import numpy as np
from car_recommender import get_car_recommender

# Initialize the recommender
@st.cache_resource
def load_recommender():
    return get_car_recommender()

# Page configuration
st.set_page_config(
    page_title="Car Recommendation System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with reduced card height and visualization styles
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .logo-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-bottom: 1rem;
        width: 100%;
    }
    .logo-img {
        max-width: 200px;
        height: auto;
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    .recommendation-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.8rem 0;
        border-left: 5px solid #1f77b4;
        min-height: auto;
    }
    .similarity-high { color: #00a86b; font-weight: bold; }
    .similarity-medium { color: #ffa500; font-weight: bold; }
    .similarity-low { color: #ff4b4b; font-weight: bold; }
    .electric-badge { 
        background-color: #00a86b; 
        color: white; 
        padding: 0.2rem 0.5rem; 
        border-radius: 12px; 
        font-size: 0.8rem;
        margin-left: 0.5rem;
    }
    .compact-text {
        margin-bottom: 0.3rem;
        line-height: 1.2;
    }
    .viz-card {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Logo and Main title - Using columns for perfect centering
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    # Logo display - perfectly centered in the middle column
    try:
        st.image("Keithston.png", width=200, use_column_width=False)
    except FileNotFoundError:
        st.warning("Logo image 'Keithston.png' not found in the current directory.")
    except Exception as e:
        st.warning(f"Could not load logo: {str(e)}")

# Main title
st.markdown('<h1 class="main-header">🚗 Smart Car Recommendation System</h1>', unsafe_allow_html=True)

# Load recommender
recommender = load_recommender()

if recommender is None:
    st.error("❌ Could not load the car recommendation system. Please check your data file.")
    st.stop()

# Sidebar for user inputs
st.sidebar.header("🎯 Your Car Preferences")

# Brand Selection
st.sidebar.subheader("🏷️ Brand Preference")
available_brands = recommender.get_available_brands()
brands = ['All Brands'] + available_brands
selected_brand = st.sidebar.selectbox("Choose your preferred brand:", brands)
brand_preference = None if selected_brand == 'All Brands' else selected_brand

# Budget
st.sidebar.subheader("💰 Budget")
budget = st.sidebar.slider(
    "Maximum Budget (₹)",
    min_value=200000,
    max_value=5000000,
    value=1500000,
    step=100000,
    format="₹%d"
)

# Fuel Type Selection
st.sidebar.subheader("⛽ Fuel Type")
fuel_type = st.sidebar.selectbox(
    "Fuel Type", 
    ['Any Fuel', 'Petrol', 'Diesel', 'Electric', 'CNG', 'Hybrid']
)
fuel_preference = None if fuel_type == 'Any Fuel' else fuel_type

# SMART LOGIC: Electric cars automatically use automatic transmission
transmission_options = ['Any Transmission', 'Manual', 'Automatic']
default_transmission = 'Any Transmission'

if fuel_type == 'Electric':
    default_transmission = 'Automatic'
    st.sidebar.info("⚡ Electric cars automatically use automatic transmission")

# Transmission selection
transmission = st.sidebar.selectbox(
    "Transmission",
    transmission_options,
    index=transmission_options.index(default_transmission)
)
transmission_preference = None if transmission == 'Any Transmission' else transmission

# Efficiency/Range based on fuel type
st.sidebar.subheader("📊 Efficiency Preferences")

if fuel_type == 'Electric':
    efficiency = st.sidebar.slider(
        "Desired Range (km per charge)", 
        min_value=100, 
        max_value=600, 
        value=300, 
        step=10
    )
else:
    efficiency = st.sidebar.slider(
        "Desired Mileage (km/l)", 
        min_value=10, 
        max_value=30, 
        value=18
    )

# Engine specifications
st.sidebar.subheader("🔧 Engine Specifications")

if fuel_type == 'Electric':
    st.sidebar.info("⚡ Electric cars don't have traditional engine displacement")
    displacement = 0
else:
    displacement = st.sidebar.slider(
        "Engine Size (cc)", 
        min_value=800, 
        max_value=3000, 
        value=1500
    )

# Features
st.sidebar.subheader("⚙️ Must-Have Features")
abs_feature = st.sidebar.checkbox("ABS", value=True)
power_steering = st.sidebar.checkbox("Power Steering", value=True)
power_windows = st.sidebar.checkbox("Power Windows", value=True)
airbags = st.sidebar.checkbox("Airbags", value=True)
ac = st.sidebar.checkbox("Air Conditioning", value=True)

# Additional Preferences
st.sidebar.subheader("🎛️ Additional Preferences")

body_type = st.sidebar.selectbox(
    "Body Type",
    ['Any Type', 'SUV', 'Sedan', 'Hatchback', 'MUV', 'Coupe', 'Convertible']
)
body_preference = None if body_type == 'Any Type' else body_type

seating = st.sidebar.selectbox(
    "Seating Capacity",
    ['Any Seating', '5', '7', '8']
)
seating_preference = None if seating == 'Any Seating' else int(seating)

# Prepare user preferences
user_preferences = [
    budget,
    efficiency,
    displacement,
    1 if abs_feature else 0,
    1 if power_steering else 0,
    1 if power_windows else 0,
    1 if airbags else 0,
    1 if ac else 0
]

# Display current preferences summary
st.sidebar.markdown("---")
st.sidebar.subheader("📋 Your Preferences Summary")
if fuel_preference:
    st.sidebar.write(f"**Fuel:** {fuel_preference}")
if body_preference:
    st.sidebar.write(f"**Body Type:** {body_preference}")
if brand_preference:
    st.sidebar.write(f"**Brand:** {brand_preference}")
st.sidebar.write(f"**Budget:** ₹{budget:,}")
if fuel_type == 'Electric':
    st.sidebar.write(f"**Range:** {efficiency} km")
    st.sidebar.write("**Transmission:** Automatic (Auto-set for Electric)")
else:
    st.sidebar.write(f"**Mileage:** {efficiency} km/l")
    if transmission_preference:
        st.sidebar.write(f"**Transmission:** {transmission_preference}")

# Main content area
st.markdown("### Find Your Perfect Car Match")

# Recommendation button
if st.button("🚀 Find My Perfect Car", use_container_width=True):
    
    if not any([abs_feature, power_steering, power_windows, airbags, ac]):
        st.warning("⚠️ Please select at least one feature you want in your car!")
    else:
        with st.spinner('🔍 Searching for the best cars matching your preferences...'):
            try:
                recommendations = recommender.recommend_cars_enhanced(
                    user_preferences,
                    brand_preference=brand_preference,
                    body_type_preference=body_preference,
                    fuel_type_preference=fuel_preference,
                    transmission_preference=transmission_preference,
                    seating_preference=seating_preference,
                    max_price=budget,
                    n_recommendations=5
                )
                
                if recommendations is not None and len(recommendations) > 0:
                    st.success(f"🎉 Found {len(recommendations)} cars matching your preferences!")
                    
                    # Create tabs for different views
                    tab1, tab2, tab3 = st.tabs(["📋 Car List", "📊 Specifications Comparison", "📈 Market Analysis"])
                    
                    with tab1:
                        # Display recommendations with compact layout
                        for idx, car in recommendations.iterrows():
                            info = recommender.get_car_display_info(car)
                            
                            with st.container():
                                st.markdown('<div class="recommendation-card">', unsafe_allow_html=True)
                                
                                col1, col2, col3 = st.columns([2, 2, 1])
                                
                                with col1:
                                    make_display = f"{info['make']}"
                                    if info['fuel_type'] == 'Electric':
                                        make_display += '<span class="electric-badge">⚡ Electric</span>'
                                    st.markdown(f"**{make_display} {info['model']}**", unsafe_allow_html=True)
                                    st.markdown(f'<p class="compact-text"><strong>Variant:</strong> {info["variant"]}</p>', unsafe_allow_html=True)
                                    st.markdown(f'<p class="compact-text"><strong>Price:</strong> {info["price"]}</p>', unsafe_allow_html=True)
                                
                                with col2:
                                    st.markdown(f'<p class="compact-text"><strong>Fuel Type:</strong> {info["fuel_type"]}</p>', unsafe_allow_html=True)
                                    st.markdown(f'<p class="compact-text"><strong>Transmission:</strong> {info["transmission"]}</p>', unsafe_allow_html=True)
                                    st.markdown(f'<p class="compact-text"><strong>Body Type:</strong> {info["body_type"]}</p>', unsafe_allow_html=True)
                                    st.markdown(f'<p class="compact-text"><strong>Seats:</strong> {info["seating_capacity"]}</p>', unsafe_allow_html=True)
                                    
                                    if info['fuel_type'] == 'Electric':
                                        if info['efficiency'] != 'N/A':
                                            st.markdown(f'<p class="compact-text"><strong>Range:</strong> {info["efficiency"]}</p>', unsafe_allow_html=True)
                                        else:
                                            st.markdown(f'<p class="compact-text"><strong>Range:</strong> Not specified</p>', unsafe_allow_html=True)
                                    else:
                                        st.markdown(f'<p class="compact-text"><strong>Efficiency:</strong> {info["efficiency"]}</p>', unsafe_allow_html=True)
                                
                                with col3:
                                    similarity = info['similarity_percent']
                                    if similarity > 90:
                                        st.markdown(f'<p class="similarity-high">Match: {similarity:.1f}%</p>', unsafe_allow_html=True)
                                    elif similarity > 70:
                                        st.markdown(f'<p class="similarity-medium">Match: {similarity:.1f}%</p>', unsafe_allow_html=True)
                                    else:
                                        st.markdown(f'<p class="similarity-low">Match: {similarity:.1f}%</p>', unsafe_allow_html=True)
                                
                                st.markdown('</div>', unsafe_allow_html=True)
                    
                    with tab2:
                        st.markdown("### 📊 Car Specifications Comparison")
                        
                        # Prepare data for visualizations - FIXED DUPLICATE ISSUE
                        car_data = []
                        
                        for idx, car in recommendations.iterrows():
                            info = recommender.get_car_display_info(car)
                            
                            # Create unique car identifier with variant
                            car_identifier = f"{info['make']} {info['model']} - {info['variant'][:30]}..."
                            
                            # Extract numeric price
                            price_str = info['price'].replace('₹', '').replace(',', '').split()[0]
                            try:
                                price_value = float(price_str)
                            except:
                                price_value = 0
                            
                            # Extract numeric efficiency
                            efficiency_value = 0
                            if info['efficiency'] != 'N/A':
                                eff_str = str(info['efficiency']).split()[0]
                                try:
                                    efficiency_value = float(eff_str)
                                except:
                                    efficiency_value = 0
                            
                            car_data.append({
                                'Car': car_identifier,
                                'Make': info['make'],
                                'Model': info['model'],
                                'Variant': info['variant'],
                                'Price': price_value,
                                'Efficiency': efficiency_value,
                                'Similarity': info['similarity_percent'],
                                'Fuel Type': info['fuel_type'],
                                'Transmission': info['transmission'],
                                'Body Type': info['body_type'],
                                'Seats': info['seating_capacity']
                            })
                        
                        # Create DataFrame from collected data
                        comparison_df = pd.DataFrame(car_data)
                        
                        # Create comparison charts
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown('<div class="viz-card">', unsafe_allow_html=True)
                            st.markdown("#### 💰 Price Comparison")
                            
                            if len(comparison_df) > 0:
                                # Use simplified car names for better display
                                display_names = comparison_df['Make'] + ' ' + comparison_df['Model']
                                
                                price_chart_data = pd.DataFrame({
                                    'Car': display_names,
                                    'Price (₹ Lakhs)': comparison_df['Price'] / 100000
                                })
                                
                                st.bar_chart(price_chart_data.set_index('Car'), use_container_width=True)
                                
                                # Show actual prices
                                st.markdown("**Actual Prices:**")
                                for _, row in comparison_df.iterrows():
                                    st.write(f"- {row['Make']} {row['Model']}: ₹{row['Price']:,.0f}")
                            else:
                                st.info("No data available for price comparison")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown('<div class="viz-card">', unsafe_allow_html=True)
                            st.markdown("#### ⚡ Efficiency Comparison")
                            
                            if len(comparison_df) > 0:
                                display_names = comparison_df['Make'] + ' ' + comparison_df['Model']
                                
                                eff_chart_data = pd.DataFrame({
                                    'Car': display_names,
                                    'Efficiency': comparison_df['Efficiency']
                                })
                                
                                st.bar_chart(eff_chart_data.set_index('Car'), use_container_width=True)
                                
                                # Show actual efficiency values
                                st.markdown("**Actual Efficiency/Range:**")
                                for _, row in comparison_df.iterrows():
                                    unit = "km" if row['Fuel Type'] == 'Electric' else "km/l"
                                    st.write(f"- {row['Make']} {row['Model']}: {row['Efficiency']} {unit}")
                            else:
                                st.info("No data available for efficiency comparison")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Similarity scores and detailed table
                        st.markdown('<div class="viz-card">', unsafe_allow_html=True)
                        st.markdown("#### 🎯 Match Scores & Specifications")
                        
                        if len(comparison_df) > 0:
                            # Create display table
                            display_table = comparison_df[['Make', 'Model', 'Price', 'Efficiency', 'Similarity', 'Fuel Type', 'Transmission']].copy()
                            display_table['Price'] = display_table['Price'].apply(lambda x: f"₹{x:,.0f}")
                            display_table['Efficiency'] = display_table.apply(
                                lambda row: f"{row['Efficiency']} {'km' if row['Fuel Type'] == 'Electric' else 'km/l'}", 
                                axis=1
                            )
                            display_table['Similarity'] = display_table['Similarity'].apply(lambda x: f"{x:.1f}%")
                            display_table = display_table.rename(columns={
                                'Make': 'Brand',
                                'Model': 'Model',
                                'Price': 'Price',
                                'Efficiency': 'Efficiency/Range',
                                'Similarity': 'Match Score',
                                'Fuel Type': 'Fuel',
                                'Transmission': 'Transmission'
                            })
                            
                            st.dataframe(display_table, use_container_width=True, hide_index=True)
                            
                            st.markdown(f"**Total cars compared: {len(comparison_df)}**")
                        else:
                            st.info("No data available for comparison")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with tab3:
                        st.markdown("### 📈 Market Analysis")
                        
                        if len(comparison_df) > 0:
                            # Fuel type distribution
                            fuel_counts = comparison_df['Fuel Type'].value_counts()
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown('<div class="viz-card">', unsafe_allow_html=True)
                                st.markdown("#### ⛽ Fuel Type Distribution")
                                
                                fuel_data = pd.DataFrame({
                                    'Fuel Type': fuel_counts.index,
                                    'Count': fuel_counts.values
                                })
                                
                                st.bar_chart(fuel_data.set_index('Fuel Type'), use_container_width=True)
                                st.markdown('</div>', unsafe_allow_html=True)
                            
                            with col2:
                                st.markdown('<div class="viz-card">', unsafe_allow_html=True)
                                st.markdown("#### 💸 Price Range Analysis")
                                
                                # Price distribution
                                price_ranges = ['Budget (<10L)', 'Mid-Range (10-20L)', 'Premium (20L+)']
                                price_counts = [0, 0, 0]
                                
                                for price in comparison_df['Price']:
                                    if price < 1000000:
                                        price_counts[0] += 1
                                    elif price < 2000000:
                                        price_counts[1] += 1
                                    else:
                                        price_counts[2] += 1
                                
                                price_dist_data = pd.DataFrame({
                                    'Price Range': price_ranges,
                                    'Count': price_counts
                                })
                                
                                st.bar_chart(price_dist_data.set_index('Price Range'), use_container_width=True)
                                st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Key metrics summary
                            st.markdown("#### 📊 Quick Stats")
                            metric_cols = st.columns(4)
                            
                            with metric_cols[0]:
                                avg_price = comparison_df['Price'].mean()
                                st.metric("Average Price", f"₹{avg_price:,.0f}")
                            
                            with metric_cols[1]:
                                avg_efficiency = comparison_df['Efficiency'].mean()
                                st.metric("Avg Efficiency", f"{avg_efficiency:.1f}")
                            
                            with metric_cols[2]:
                                avg_similarity = comparison_df['Similarity'].mean()
                                st.metric("Avg Match Score", f"{avg_similarity:.1f}%")
                            
                            with metric_cols[3]:
                                best_match = comparison_df['Similarity'].max()
                                st.metric("Best Match", f"{best_match:.1f}%")
                            
                            # Show which car has the best match
                            best_car = comparison_df.loc[comparison_df['Similarity'].idxmax()]
                            st.info(f"🏆 **Best Match**: {best_car['Make']} {best_car['Model']} with {best_car['Similarity']:.1f}% similarity")
                            
                        else:
                            st.info("No data available for market analysis")
                
                else:
                    st.error("❌ No cars found matching your criteria. Try adjusting your preferences!")
                    
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")

# Information section
st.markdown("---")
st.markdown("### 💡 How It Works")

col1, col2, col3 = st.columns(3)

with col1:
    st.info("""
    **🎯 Smart Matching**
    - Uses machine learning to find similar cars
    - Considers 30+ features and specifications
    - Ranks by similarity score
    """)

with col2:
    st.info("""
    **⚡ Smart Features**
    - Electric car range tracking
    - Automatic transmission for EVs
    - Budget-aware recommendations
    - Context-aware efficiency metrics
    """)

with col3:
    st.info("""
    **🚗 Comprehensive Database**
    - 1,200+ car variants
    - Multiple brands and body types
    - Electric and fuel cars
    - Real specifications and prices
    """)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>Built with ❤️ using Streamlit and Machine Learning</p>", 
    unsafe_allow_html=True
)