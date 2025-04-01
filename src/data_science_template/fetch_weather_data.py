"""Script to fetch weather data for Burlington, VT using Open-Meteo API."""

import requests
import pandas as pd
from datetime import datetime
from pathlib import Path

def get_weather_data(lat: float, lon: float, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Fetch weather data from Open-Meteo API.
    
    Args:
        lat: Latitude
        lon: Longitude
        start_date: Start date for data
        end_date: End date for data
        
    Returns:
        DataFrame containing weather data
    """
    # Open-Meteo API endpoint
    base_url = "https://archive-api.open-meteo.com/v1/archive"
    
    # Format dates for API
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_str,
        "end_date": end_str,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "pressure_msl",
            "windspeed_10m",
            "precipitation"
        ],
        "timezone": "America/New_York"
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Extract hourly data
        hourly_data = data["hourly"]
        
        # Create DataFrame
        df = pd.DataFrame({
            "date": pd.to_datetime(hourly_data["time"]),
            "temperature": hourly_data["temperature_2m"],
            "humidity": hourly_data["relative_humidity_2m"],
            "pressure": hourly_data["pressure_msl"],
            "wind_speed": hourly_data["windspeed_10m"],
            "rainfall": hourly_data["precipitation"]
        })
        
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()

def main():
    """Main function to fetch and save weather data."""
    # Burlington, VT coordinates
    BURLINGTON_LAT = 44.4759
    BURLINGTON_LON = -73.2121
    
    # Date range
    start_date = datetime(2025, 1, 1)
    end_date = datetime(2025, 3, 31)
    
    # Create data directory if it doesn't exist
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Fetch data
    print("Fetching weather data for Burlington, VT...")
    df = get_weather_data(BURLINGTON_LAT, BURLINGTON_LON, start_date, end_date)
    
    if not df.empty:
        # Save to CSV
        output_path = data_dir / "weather_data.csv"
        df.to_csv(output_path, index=False)
        print(f"Data saved to {output_path}")
        print(f"Retrieved {len(df)} data points")
    else:
        print("No data was retrieved")

if __name__ == "__main__":
    main() 