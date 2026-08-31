# --------------------------------------------------------------------------------
# Weather Data Analysis for Location Entered by User
# --------------------------------------------------------------------------------
import os
import json
import requests                             # uv add requests
import pandas as pd                         # uv add pandas
import matplotlib.pyplot as plt             # uv add matplotlib
from openai import OpenAI                   # uv add openai
from dotenv import load_dotenv              # uv add --dev python-dotenv
from datetime import datetime, timedelta

# --------------------------------------------------------------------------------
# OpenAI API Parameters
# --------------------------------------------------------------------------------
load_dotenv()
token = os.environ["GITHUB_TOKEN"]
endpoint = "https://models.github.ai/inference"
model = "openai/gpt-4.1-nano"

# --------------------------------------------------------------------------------
# 1b.Validate user input for location (OpenAI API : 1st Call)
# --------------------------------------------------------------------------------
def validate_location(location_name: str) -> bool:
    client = OpenAI(base_url=endpoint, api_key=token)
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Determine if this is a geographical location, returning either 'True' or 'False'."},
            {"role": "user", "content": location_name}
        ]
    )

    response = eval(completion.choices[0].message.content)
    return response

# --------------------------------------------------------------------------------
# 1a. Get user input for location
# --------------------------------------------------------------------------------
def input_location() -> str:
    while True:
        location_name = input("\nPlease enter a location: ")
        location_is_valid = validate_location(location_name)

        if location_is_valid:
            print(f"'{location_name}' is a valid location.")
            break
        else:
            print(f"'{location_name}' is NOT a valid location.")
            continue

    return location_name

# --------------------------------------------------------------------------------
# 2. Get location coordinates (OpenAI API : 2nd Call)
# --------------------------------------------------------------------------------
def get_location_coordinates(location_name: str) -> tuple:
    system_prompt = "Determine the latitude and longitude of this location, returning a JSON object with 'latitude' and 'longitude' keys."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": location_name}
    ]

    client = OpenAI(
        base_url=endpoint,
        api_key=token,
    )

    completion = client.chat.completions.create(
        model=model,
        messages=messages
    )

    location_latitude = json.loads(completion.choices[0].message.content)["latitude"]
    location_longitude = json.loads(completion.choices[0].message.content)["longitude"]

    return location_latitude, location_longitude

# --------------------------------------------------------------------------------
# 3. Get user input for days
# --------------------------------------------------------------------------------
def input_days() -> int:
    while True:
        days = input("\nPlease enter number of days: ")

        try:
            days = int(days)
        except:
            print(f"'{days}' is NOT a valid number of days.")
            continue
        else:
            print(f"'{days}' is a valid number of days.")
            break

    return days

# --------------------------------------------------------------------------------
# 4. Get date range
# --------------------------------------------------------------------------------
def get_date_range(days: int) -> tuple:
    today = datetime.now()
    days_ago = today - timedelta(days=days)
    start_date = days_ago.strftime("%Y-%m-%d")
    end_date = today.strftime("%Y-%m-%d")

    return start_date, end_date

# --------------------------------------------------------------------------------
# 5. Get weather data (Open-Meteo Weather API)
# --------------------------------------------------------------------------------
def get_weather_data(location_latitude: float, location_longitude: float, start_date: str, end_date: str) -> dict:
    url = f"https://api.open-meteo.com/v1/forecast?latitude={location_latitude}&longitude={location_longitude}&start_date={start_date}&end_date={end_date}&daily=temperature_2m_max,temperature_2m_min"
    response = requests.get(url)
    data = response.json()

    return data

# --------------------------------------------------------------------------------
# 6. Process weather data (pandas)
# --------------------------------------------------------------------------------
def process_weather_data(data: dict, location_name: str, start_date: str, end_date: str) -> pd.DataFrame:
    df = pd.DataFrame({
        'date': pd.to_datetime(data['daily']['time']),
        'max_temp': data['daily']['temperature_2m_max'],
        'min_temp': data['daily']['temperature_2m_min']
    })

    return df

# --------------------------------------------------------------------------------
# 7. Calculate average (pandas)
# --------------------------------------------------------------------------------
def calculate_average_temperature(df: pd.DataFrame) -> pd.DataFrame:
    df['avg_temp'] = (df['max_temp'] + df['min_temp']) / 2
    return df

# --------------------------------------------------------------------------------
# 8. Create visualisation (matplotlib)
# --------------------------------------------------------------------------------
def create_visualisation(df: pd.DataFrame, location_name: str, days: int) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(df['date'], df['max_temp'], 'r-o', label='Max')
    plt.plot(df['date'], df['min_temp'], 'b-o', label='Min')
    plt.plot(df['date'], df['avg_temp'], 'g--', label='Average')

    plt.xlabel('Date')
    plt.ylabel('Temperature (°C)')
    plt.title(f'{location_name} Weather - Past {days} Days')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

# --------------------------------------------------------------------------------
# 9. Save data (pandas) and visualisation (matplotlib)
# --------------------------------------------------------------------------------
def save_data(df: pd.DataFrame, location_name: str, start_date: str, end_date: str) -> None:
    if not os.path.exists('data'):
        os.makedirs('data')

    filename = f'weather_{location_name.replace(" ", "_").lower()}_{start_date}_{end_date}'

    plt.savefig(f'data/{filename}.png')
    df.to_csv(f'data/{filename}.csv', index=False)

    print("\nFiles saved in 'data' folder")
    print(f"data/{filename}.png")
    print(f"data/{filename}.csv")

# --------------------------------------------------------------------------------
# 10. Temperature Statistics (pandas)
# --------------------------------------------------------------------------------
def temperatures_statistics(df: pd.DataFrame) -> None:

    print(f"\nDaytime temperature:")
    print(f"Maximum: {df['max_temp'].max():.1f}°C")
    print(f"Minimum: {df['max_temp'].min():.1f}°C")
    print(f"Average: {df['max_temp'].mean():.1f}°C")

    print(f"\nNighttime temperature:")
    print(f"Maximum: {df['min_temp'].max():.1f}°C")
    print(f"Minimum: {df['min_temp'].min():.1f}°C")
    print(f"Average: {df['min_temp'].mean():.1f}°C")

    print(f"\nAverage temperature:")
    print(f"Maximum: {df['avg_temp'].max():.1f}°C")
    print(f"Minimum: {df['avg_temp'].min():.1f}°C")
    print(f"Average: {df['avg_temp'].mean():.1f}°C")

# --------------------------------------------------------------------------------
# Main Function
# --------------------------------------------------------------------------------
def main():

    # 1. Get and validate user input for location (OpenAI API : 1st Call)
    location_name = input_location()

    # 2. Get location coordinates (OpenAI API : 2nd Call)
    location_latitude, location_longitude = get_location_coordinates(location_name)
    print(f"\nLocation: {location_name}, Latitude: {location_latitude}, Longitude: {location_longitude}")

    # 3. Get user input for days
    days = input_days()

    # 4. Get date range
    start_date, end_date = get_date_range(days)
    print(f"\nDays: {days}, Date Range: {start_date} to {end_date}")

    # 5. Get weather data (Open-Meteo Weather API)
    data = get_weather_data(location_latitude, location_longitude, start_date, end_date)

    # 6. Process weather data (pandas)
    df = process_weather_data(data, location_name, start_date, end_date)

    # 7. Calculate average temperature (pandas)
    df = calculate_average_temperature(df)

    # 8. Create visualisation (matplotlib)
    create_visualisation(df, location_name, days)

    # 9. Save data (pandas) and visualisation (matplotlib)
    save_data(df, location_name, start_date, end_date)

    # 10. Temperature Statistics (pandas)
    temperatures_statistics(df)


if __name__ == "__main__":
    main()
