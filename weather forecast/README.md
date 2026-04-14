# 🌦️ Weather Forecast App

A Streamlit-based web application that displays current weather and 7-day forecasts using OpenWeatherMap's One Call API 3.0.

## Features

- 🌡️ Current weather conditions (temperature, humidity, wind speed, description)
- 📅 7-day weather forecast
- 🌍 Search for any city worldwide
- 🎨 Clean and intuitive UI with emojis

## Prerequisites

- Python 3.7 or higher
- OpenWeatherMap API key (already included in the app)

## Installation

1. **Clone or download this repository**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   streamlit run app.py
   ```

4. **Access the app:**
   - The app will automatically open in your default web browser
   - Or navigate to: `http://localhost:8501`

## Usage

1. Enter a city name in the text input field (default: Delhi)
2. Click the "Get Weather" button
3. View current weather conditions and 7-day forecast

## Project Structure

```
weather forecast/
├── app.py              # Main application file
├── requirements.txt    # Python dependencies
└── README.md          # Documentation
```

## Technologies Used

- **Python** - Programming language
- **Streamlit** - Web application framework
- **Requests** - HTTP library for API calls
- **OpenWeatherMap API** - Weather data provider

## API Information

This app uses OpenWeatherMap's One Call API 3.0 which provides:
- Current weather data
- Minute forecast
- Hourly forecast
- Daily forecast
- Weather alerts

## Author

Created with ❤️ using Streamlit and OpenWeatherMap API

## License

This project is open source and available under the MIT License.
