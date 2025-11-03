import os
import sys
import subprocess

def main():
    # Check if data file exists
    data_path = "data/cars_ds_final.csv"
    if not os.path.exists(data_path):
        print(f"❌ Error: Data file not found at {data_path}")
        print("Please make sure your CSV file is in the data/ directory")
        return
    
    print("🚗 Starting Car Recommendation App...")
    print("📊 This will open in your web browser...")
    
    # Get the path to the app
    app_path = os.path.join("src", "app.py")
    
    # Run streamlit command
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", app_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running Streamlit: {e}")
    except FileNotFoundError:
        print("❌ Streamlit not found. Please install with: pip install streamlit")
    except KeyboardInterrupt:
        print("\n👋 App closed by user")

if __name__ == "__main__":
    main()