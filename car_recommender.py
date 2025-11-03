# car_recommender.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import os
import warnings
import re

# Suppress the specific warning about feature names
warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names")

class CarRecommender:
    def __init__(self, data_path=None):
        self.df = None
        self.feature_columns = []
        self.feature_names = []
        self.scaler = None
        self.knn_model = None
        
        # Set default data path if not provided
        if data_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            data_path = os.path.join(current_dir, 'data', 'cars_ds_final.csv')
        
        self.load_and_prepare_data(data_path)
    
    def load_and_prepare_data(self, data_path):
        """Load and prepare the car dataset"""
        try:
            print(f"Looking for data at: {data_path}")
            
            # Check if file exists
            if not os.path.exists(data_path):
                # Try alternative paths
                alternative_paths = [
                    'cars_ds_final.csv',
                    './cars_ds_final.csv',
                    '../cars_ds_final.csv',
                    'data/cars_ds_final.csv',
                    './data/cars_ds_final.csv'
                ]
                
                for alt_path in alternative_paths:
                    if os.path.exists(alt_path):
                        data_path = alt_path
                        print(f"Found data at: {data_path}")
                        break
                else:
                    raise FileNotFoundError(f"Could not find data file. Tried: {data_path} and {alternative_paths}")
            
            # Load data
            self.df = pd.read_csv(data_path)
            print(f"✅ Successfully loaded data with {len(self.df)} cars")
            
            # Apply all the cleaning and feature engineering
            self._clean_data()
            self._create_features()
            self._build_model()
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            raise
    
    def _clean_data(self):
        """Clean the dataset"""
        print("🧹 Cleaning data...")
        
        # Clean Make column - convert all to string and handle missing values
        self.df['Make'] = self.df['Make'].fillna('Unknown').astype(str)
        
        # Clean other string columns that might be used in the app
        if 'Model' in self.df.columns:
            self.df['Model'] = self.df['Model'].fillna('Unknown').astype(str)
        if 'Variant' in self.df.columns:
            self.df['Variant'] = self.df['Variant'].fillna('Unknown').astype(str)
        
        # Clean Fuel Type - use the original Fuel_Type column
        if 'Fuel_Type' in self.df.columns:
            self.df['Fuel_Type'] = self.df['Fuel_Type'].fillna('Unknown').astype(str)
            # Create cleaned fuel type
            self.df['Fuel_Type_Cleaned'] = self.df['Fuel_Type'].apply(self._clean_fuel_type)
        else:
            self.df['Fuel_Type_Cleaned'] = 'Unknown'
        
        # Clean Transmission - use the Type column
        if 'Type' in self.df.columns:
            self.df['Type'] = self.df['Type'].fillna('Unknown').astype(str)
            # Create cleaned transmission
            self.df['Transmission_Cleaned'] = self.df['Type'].apply(self._clean_transmission)
        else:
            self.df['Transmission_Cleaned'] = 'Unknown'
        
        # Clean Seating Capacity
        if 'Seating_Capacity' in self.df.columns:
            self.df['Seating_Capacity_Cleaned'] = self.df['Seating_Capacity'].apply(self._clean_seating_capacity)
        else:
            self.df['Seating_Capacity_Cleaned'] = 'N/A'
        
        # Clean Body Type
        if 'Body_Type' in self.df.columns:
            self.df['Body_Type'] = self.df['Body_Type'].fillna('Unknown').astype(str)
            self.df['Body_Type_Cleaned'] = self.df['Body_Type'].apply(self._clean_body_type)
        else:
            self.df['Body_Type_Cleaned'] = 'Unknown'
        
        # Price cleaning
        self.df['Price_Cleaned'] = self.df['Ex-Showroom_Price'].apply(self._clean_price)
        
        # Mileage cleaning
        self.df['Mileage_Cleaned'] = self.df['ARAI_Certified_Mileage'].apply(self._clean_mileage)
        
        # Displacement cleaning
        self.df['Displacement_Cleaned'] = self.df['Displacement'].apply(self._clean_displacement)
        
        # Fill missing values for numerical columns
        self.df['Mileage_Cleaned'].fillna(self.df['Mileage_Cleaned'].median(), inplace=True)
        self.df['Displacement_Cleaned'].fillna(self.df['Displacement_Cleaned'].median(), inplace=True)
        
        print("✅ Data cleaning complete")
    
    def _create_features(self):
        """Create all the features"""
        print("🔧 Creating features...")
        
        # Binary features
        self.df['Has_ABS'] = self.df['ABS_(Anti-lock_Braking_System)'].apply(lambda x: 1 if x == 'Yes' else 0)
        self.df['Has_Power_Steering'] = self.df['Power_Steering'].apply(lambda x: 1 if pd.notna(x) and str(x).strip() != '' else 0)
        self.df['Has_Power_Windows'] = self.df['Power_Windows'].apply(lambda x: 1 if pd.notna(x) and str(x).strip() != '' else 0)
        self.df['Has_Airbags'] = self.df['Number_of_Airbags'].apply(lambda x: 1 if pd.notna(x) and float(x) > 0 else 0)
        
        # AC feature
        if 'Second_Row_AC_Vents' in self.df.columns:
            self.df['Has_AC'] = self.df['Second_Row_AC_Vents'].apply(lambda x: 1 if pd.notna(x) and str(x).strip() != '' else 0)
        else:
            self.df['Has_AC'] = 1
        
        # Body type features
        body_dummies = pd.get_dummies(self.df['Body_Type_Cleaned'], prefix='Body')
        self.df = pd.concat([self.df, body_dummies], axis=1)
        
        # Define feature columns
        self.feature_columns = [
            'Price_Cleaned', 'Mileage_Cleaned', 'Displacement_Cleaned',
            'Has_ABS', 'Has_Power_Steering', 'Has_Power_Windows', 
            'Has_Airbags', 'Has_AC'
        ]
        
        # Add body type columns
        body_cols = [col for col in self.df.columns if col.startswith('Body_') and col not in ['Body_Type', 'Body_Type_Cleaned']]
        self.feature_columns.extend(body_cols)
        
        print(f"✅ Created {len(self.feature_columns)} features")
    
    def _build_model(self):
        """Build the KNN model"""
        print("🤖 Building ML model...")
        
        feature_df = self.df[self.feature_columns].copy()
        feature_df.fillna(0, inplace=True)
        
        # Store feature names for later use
        self.feature_names = feature_df.columns.tolist()
        
        self.scaler = StandardScaler()
        scaled_features = self.scaler.fit_transform(feature_df)
        
        self.knn_model = NearestNeighbors(n_neighbors=10, metric='cosine')
        self.knn_model.fit(scaled_features)
        
        print("✅ Model training complete")
    
    def get_available_brands(self):
        """Get list of available car brands"""
        try:
            # Ensure all values are strings and clean them
            brands = self.df['Make'].fillna('Unknown').astype(str).unique().tolist()
            
            # Remove any empty strings or problematic values
            brands = [brand for brand in brands if brand and brand != 'nan' and brand != 'Unknown']
            
            # Sort case-insensitively
            return sorted(brands, key=str.lower)
            
        except Exception as e:
            print(f"Error getting brands: {e}")
            # Return some common Indian car brands as fallback
            return ['Tata', 'Hyundai', 'Maruti Suzuki', 'Honda', 'Toyota', 'Mahindra', 'Kia', 'MG', 'Renault', 'Ford']
    
    def recommend_cars_enhanced(self, user_preferences, brand_preference=None, 
                              body_type_preference=None, fuel_type_preference=None, 
                              transmission_preference=None, seating_preference=None,
                              max_price=None, n_recommendations=5):
        """Main recommendation function"""
        try:
            # Apply filters
            filtered_df = self.df.copy()
            
            # Brand filter
            if brand_preference:
                filtered_df = filtered_df[filtered_df['Make'].str.lower() == brand_preference.lower()]
            
            # Price filter
            if max_price:
                filtered_df = filtered_df[filtered_df['Price_Cleaned'] <= max_price]
            
            # Body type filter
            if body_type_preference:
                body_col = f'Body_{body_type_preference}'
                if body_col in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df[body_col] == 1]
            
            # Fuel type filter - FIXED
            if fuel_type_preference:
                filtered_df = filtered_df[filtered_df['Fuel_Type_Cleaned'].str.lower() == fuel_type_preference.lower()]
            
            # Transmission filter - FIXED
            if transmission_preference:
                filtered_df = filtered_df[filtered_df['Transmission_Cleaned'].str.lower() == transmission_preference.lower()]
            
            # Seating capacity filter - FIXED
            if seating_preference:
                seating_str = str(seating_preference)
                filtered_df = filtered_df[filtered_df['Seating_Capacity_Cleaned'] == seating_str]
            
            if len(filtered_df) == 0:
                print("❌ No cars found matching the filters")
                return None
            
            # Prepare features for the filtered dataset
            filtered_features = filtered_df[self.feature_columns].copy()
            filtered_features.fillna(0, inplace=True)
            
            # Convert to DataFrame with proper feature names to avoid warnings
            filtered_features_df = pd.DataFrame(filtered_features, columns=self.feature_names)
            
            # Scale features
            filtered_features_scaled = self.scaler.transform(filtered_features_df)
            
            # Build model on filtered data
            knn_filtered = NearestNeighbors(
                n_neighbors=min(n_recommendations, len(filtered_df)), 
                metric='cosine'
            )
            knn_filtered.fit(filtered_features_scaled)
            
            # Prepare user vector as a DataFrame with proper feature names
            user_vector = self._prepare_user_vector(user_preferences)
            user_vector_df = pd.DataFrame([user_vector], columns=self.feature_names)
            user_vector_scaled = self.scaler.transform(user_vector_df)
            
            # Get recommendations
            distances, indices = knn_filtered.kneighbors(user_vector_scaled)
            
            recommendations = filtered_df.iloc[indices[0]].copy()
            recommendations['Similarity_Score'] = 1 - distances[0]
            
            print(f"✅ Found {len(recommendations)} recommendations")
            return recommendations.sort_values('Similarity_Score', ascending=False)
            
        except Exception as e:
            print(f"Error in recommendation: {e}")
            return None
    
    def _prepare_user_vector(self, user_preferences):
        """Prepare user preference vector matching feature columns"""
        base_features = user_preferences[:8]  # First 8 are basic features
        
        # Add body type preferences (zeros for all body types)
        body_cols = [col for col in self.feature_columns if col.startswith('Body_')]
        body_prefs = [0] * len(body_cols)
        
        return base_features + body_prefs
    
    def get_car_display_info(self, car):
        """Extract display information for a car"""
        fuel_type = car.get('Fuel_Type_Cleaned', 'Unknown')
        
        # Handle efficiency display based on fuel type
        if fuel_type == 'Electric':
            # For electric cars, try to get electric range first
            electric_range = car.get('Electric_Range')
            if pd.notna(electric_range) and electric_range != '':
                efficiency_display = f"{electric_range} km"
            else:
                # Fallback to mileage if electric range not available
                mileage = car.get('Mileage_Cleaned', 'N/A')
                efficiency_display = f"{mileage} km" if mileage != 'N/A' else 'Range not specified'
        else:
            # For fuel cars, show mileage
            mileage = car.get('Mileage_Cleaned', 'N/A')
            efficiency_display = f"{mileage} km/l" if mileage != 'N/A' else 'Efficiency not specified'
        
        return {
            'make': car['Make'] if pd.notna(car['Make']) else 'Unknown',
            'model': car['Model'] if pd.notna(car['Model']) else 'Unknown',
            'variant': car['Variant'] if pd.notna(car['Variant']) else 'Unknown',
            'price': f"₹{car['Price_Cleaned']:,.0f}" if pd.notna(car['Price_Cleaned']) else 'Price not available',
            'fuel_type': fuel_type,
            'transmission': car.get('Transmission_Cleaned', 'Unknown'),
            'body_type': car.get('Body_Type_Cleaned', 'Unknown'),
            'seating_capacity': f"{car.get('Seating_Capacity_Cleaned', 'N/A')} seats",
            'efficiency': efficiency_display,
            'similarity_percent': (car.get('Similarity_Score', 0) * 100)
        }
    
    # Helper cleaning functions
    def _clean_price(self, price_str):
        if pd.isna(price_str):
            return np.nan
        cleaned = str(price_str).replace('Rs. ', '').replace(',', '')
        try:
            return float(cleaned)
        except:
            return np.nan
    
    def _clean_mileage(self, mileage_str):
        if pd.isna(mileage_str) or mileage_str == '?':
            return np.nan
        if isinstance(mileage_str, str):
            numbers = re.findall(r'\d+\.?\d*', str(mileage_str))
            if numbers:
                return float(numbers[0])
        return np.nan
    
    def _clean_displacement(self, disp_str):
        if pd.isna(disp_str):
            return np.nan
        if isinstance(disp_str, str):
            numbers = re.findall(r'\d+', str(disp_str))
            if numbers:
                return int(numbers[0])
        return np.nan
    
    def _clean_body_type(self, body_type):
        if pd.isna(body_type):
            return 'Unknown'
        body_type = str(body_type).lower()
        if 'suv' in body_type:
            return 'SUV'
        elif 'sedan' in body_type:
            return 'Sedan'
        elif 'hatchback' in body_type:
            return 'Hatchback'
        elif 'muv' in body_type or 'mpv' in body_type:
            return 'MUV'
        elif 'coupe' in body_type:
            return 'Coupe'
        elif 'convertible' in body_type:
            return 'Convertible'
        else:
            return 'Unknown'
    
    def _clean_fuel_type(self, fuel_type):
        """Clean and standardize fuel type"""
        if pd.isna(fuel_type) or fuel_type == 'Unknown':
            return 'Unknown'
        
        fuel_str = str(fuel_type).lower()
        
        if 'petrol' in fuel_str:
            return 'Petrol'
        elif 'diesel' in fuel_str:
            return 'Diesel'
        elif 'electric' in fuel_str:
            return 'Electric'
        elif 'cng' in fuel_str:
            return 'CNG'
        elif 'hybrid' in fuel_str:
            return 'Hybrid'
        else:
            return fuel_type.title()
    
    def _clean_transmission(self, transmission):
        """Clean and standardize transmission type"""
        if pd.isna(transmission) or transmission == 'Unknown':
            return 'Unknown'
        
        trans_str = str(transmission).lower()
        
        if 'automatic' in trans_str or 'amt' in trans_str or 'cvt' in trans_str or 'dct' in trans_str:
            return 'Automatic'
        elif 'manual' in trans_str:
            return 'Manual'
        else:
            return transmission.title()
    
    def _clean_seating_capacity(self, seating):
        """Clean and standardize seating capacity"""
        if pd.isna(seating):
            return 'N/A'
        
        if isinstance(seating, (int, float)):
            return f"{int(seating)}"
        elif isinstance(seating, str):
            # Extract numbers from strings like "5 Seater", "5 seats", etc.
            numbers = re.findall(r'\d+', str(seating))
            if numbers:
                return f"{numbers[0]}"
        
        return 'N/A'

def get_car_recommender():
    """Factory function to create and return recommender instance"""
    try:
        return CarRecommender()
    except Exception as e:
        print(f"Failed to initialize car recommender: {e}")
        return None