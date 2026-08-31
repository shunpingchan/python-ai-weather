# --------------------------------------------------------------------------------
# Weather Data Analysis for Location Entered by User
# --------------------------------------------------------------------------------
import os
import json
import requests
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from datetime import datetime, timedelta
from pydantic import BaseModel, ConfigDict, Field

# --------------------------------------------------------------------------------
# API Parameters
# --------------------------------------------------------------------------------
load_dotenv()
model    = "openai/gpt-4.1"

# OpenAI API Parameters
endpoint = "https://models.github.ai/inference"
token    = os.environ["GITHUB_TOKEN"]

# Requests API Parameters
API_URL  = f"{endpoint}/chat/completions"
headers  = {"Authorization": f"Bearer {token}"}

# --------------------------------------------------------------------------------
# 1c. Define the response format for structured output (pydantic)
# --------------------------------------------------------------------------------
class Location(BaseModel):
    model_config = ConfigDict(extra="forbid")

    is_valid_location: bool = Field(
        description="Whether this is a geographical location."
    )

# --------------------------------------------------------------------------------
# 1b. Validate user input for location (requests) (Requests API : 1st Call)
# --------------------------------------------------------------------------------
def validate_location(location_name: str) -> bool:

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "Location",
            "schema": Location.model_json_schema(),
            "strict": True,
        },
    }

    system_prompt = "Determine if this is a geographical location."
    user_prompt = location_name

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    payload  = {
        "messages": messages,
        "model": model,
        "response_format": response_format,
    }

    response = requests.post(API_URL, headers=headers, json=payload)

    return json.loads(response.json()["choices"][0]["message"]["content"])["is_valid_location"]

# --------------------------------------------------------------------------------
# 1a. Get user input for location
# --------------------------------------------------------------------------------
def input_location() -> str:
    while True:
        location_name = input("\nPlease enter a location: ")
        is_valid_location = validate_location(location_name)

        if is_valid_location:
            print(f"'{location_name}' is a valid location.")
            break
        else:
            print(f"'{location_name}' is NOT a valid location.")
            continue

    return location_name

# --------------------------------------------------------------------------------
# 2. Get location coordinates for location using a tool (requests) (Requests API : 2nd Call)
# --------------------------------------------------------------------------------
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather_data",
            "description": "Get temperature data for provided coordinates in celsius.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location_latitude": {"type": "number"},
                    "location_longitude": {"type": "number"},
                },
                "required": ["location_latitude", "location_longitude"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
]

def get_function(location_name: str) -> tuple:
    system_prompt = "You are a helpful weather assistant."
    user_prompt = f"Gather temperature data for {location_name}."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    payload  = {
        "messages": messages,
        "model": model,
        "tools": tools,
    }

    response = requests.post(API_URL, headers=headers, json=payload)

    func_name = response.json()["choices"][0]["message"]["tool_calls"][0]["function"]["name"]
    func_args = json.loads(response.json()["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])

    return func_name, func_args

# --------------------------------------------------------------------------------
# 3. Get and validate user input for days (try/except)
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
# 5b. Get weather data for location and date range (requests) (Open-Meteo Weather API)
# --------------------------------------------------------------------------------
def get_weather_data(start_date: str, end_date: str, location_latitude: float, location_longitude: float) -> dict:
    url = f"https://api.open-meteo.com/v1/forecast?latitude={location_latitude}&longitude={location_longitude}&start_date={start_date}&end_date={end_date}&daily=temperature_2m_max,temperature_2m_min"
    response = requests.get(url)
    data = response.json()

    return data

# --------------------------------------------------------------------------------
# 5a. Call function with location coordinates as arguments from tool
# --------------------------------------------------------------------------------
def call_function(func_name, func_args, start_date, end_date) -> dict:
    if func_name == "get_weather_data":
        return get_weather_data(start_date, end_date, **func_args)

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
# 7. Calculate average temperature (per day) (pandas)
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

    filename = f'weather_{location_name.lower()}_{start_date}_{end_date}'
    
    plt.savefig(f'data/{filename}.png')
    df.to_csv(f'data/{filename}.csv', index=False)

    print("\nFiles saved in 'data' folder:")
    print(f"data/{filename}.png")
    print(f"data/{filename}.csv")

# --------------------------------------------------------------------------------
# 10. Calculate and save statistics (min, max, average over date range) (pandas)
# --------------------------------------------------------------------------------
def temperature_statistics(df: pd.DataFrame, location_name: str, start_date: str, end_date: str) -> None:

    filename = f'weather_{location_name.lower()}_{start_date}_{end_date}'

    with open(f"data/{filename}.out", "w") as f:

        f.write(f"Daytime temperature:\n")
        f.write(f"Maximum: {df['max_temp'].max():.1f}°C\n")
        f.write(f"Minimum: {df['max_temp'].min():.1f}°C\n")
        f.write(f"Average: {df['max_temp'].mean():.1f}°C\n")

        f.write(f"\nNighttime temperature:\n")
        f.write(f"Maximum: {df['min_temp'].max():.1f}°C\n")
        f.write(f"Minimum: {df['min_temp'].min():.1f}°C\n")
        f.write(f"Average: {df['min_temp'].mean():.1f}°C\n")

        f.write(f"\nAverage temperature:\n")
        f.write(f"Maximum: {df['avg_temp'].max():.1f}°C\n")
        f.write(f"Minimum: {df['avg_temp'].min():.1f}°C\n")
        f.write(f"Average: {df['avg_temp'].mean():.1f}°C\n")

    print(f"data/{filename}.out\n")

# --------------------------------------------------------------------------------
# Main Function
# --------------------------------------------------------------------------------
def main():

    # 1. Get and validate user input for location (requests) (pydantic) (Requests API : 1st Call)
    location_name = input_location()

    # 2. Get call funcation name and arguments (location coordinates) (Requests API : 2nd Call)
    func_name, func_args = get_function(location_name)
    print(f"\nLocation: {location_name}, Function Name: {func_name}, Function Arguments: {func_args}")

    # 3. Get and validate user input for days
    days = input_days()

    # 4. Get date range
    start_date, end_date = get_date_range(days)
    print(f"\nDays: {days}, Date Range: {start_date} to {end_date}\n")

    # 5. Call funcation with arguments (location coordinates) to get weather data (Open-Meteo Weather API)
    data = call_function(func_name, func_args, start_date, end_date)
    print("Weather data retrieved.")

    # 6. Process weather data (pandas)
    df = process_weather_data(data, location_name, start_date, end_date)
    print("Processed DataFrame.")

    # 7. Calculate average temperature (per day) (pandas)
    df = calculate_average_temperature(df)
    print("Calculated average temperatures.")

    # 8. Create visualisation (matplotlib)
    create_visualisation(df, location_name, days)
    print("Calculated visualisation.")

    # 9. Save data (pandas) and visualisation (matplotlib)
    save_data(df, location_name, start_date, end_date)

    # 10. Calculate and save statistics (min, max, average over date range) (pandas)
    temperature_statistics(df, location_name, start_date, end_date)
    print("Calculated and saved data, visualisation and statistics.\n")


if __name__ == "__main__":
    main()
