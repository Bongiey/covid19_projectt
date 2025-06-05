# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from datetime import datetime
import warnings
import requests

# Replace this with your actual CSV URL
url = "https://covid.ourworldindata.org/data/owid-covid-data.csv"

# Download the file manually using requests
response = requests.get(url, stream=True)

with open("data.csv", "wb") as f:
    for chunk in response.iter_content(chunk_size=1024):
        if chunk:
            f.write(chunk)

# Now read it with pandas from the local file
df = pd.read_csv("data.csv")

print(df.head())  # Confirm it's working


# Setting visualization styles
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")
warnings.filterwarnings('ignore')

# Display settings for better readability
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

# Load the COVID-19 dataset
url = "https://covid.ourworldindata.org/data/owid-covid-data.csv"

# Show the shape of the dataset
print(f"Dataset dimensions: {df.shape}")

# Display the first few rows
df.head()

# Check the column names
print("Column names:")
print(df.columns.tolist())

# Check data types and missing values
df.info()

# Calculate summary statistics for key numeric columns
df.describe(include=[np.number])

# Check for missing values in key columns
missing_data = df[['location', 'date', 'total_cases', 'new_cases', 'total_deaths', 'new_deaths', 
                  'total_vaccinations', 'people_vaccinated', 'people_fully_vaccinated']].isnull().sum()
print("Missing values in key columns:")
print(missing_data)

# Convert date column to datetime
df['date'] = pd.to_datetime(df['date'])

# Get the latest date in the dataset
latest_date = df['date'].max()
print(f"Latest data date: {latest_date}")

# Let's focus on specific countries for our analysis
focus_countries = ['World', 'United States', 'India', 'Brazil', 'United Kingdom', 'Russia', 
                  'France', 'Germany', 'South Africa', 'Kenya', 'China', 'Japan']

# Filter data for these countries
df_countries = df[df['location'].isin(focus_countries)].copy()

# Show how many rows we have for each country
print(df_countries['location'].value_counts())

# Let's check for missing values in our focus countries
missing_by_country = df_countries.isnull().sum().groupby('location')[['total_cases', 'total_deaths', 'total_vaccinations']]
print("Missing values by country:")
print(missing_by_country)

# Get global data
df_global = df_countries[df_countries['location'] == 'World'].copy()

# Plot global cases and deaths over time
fig, ax = plt.subplots(1, 2, figsize=(20, 6))

# Total Cases
ax[0].plot(df_global['date'], df_global['total_cases'], color='steelblue', linewidth=2)
ax[0].set_title('Global COVID-19 Cases Over Time', fontsize=16)
ax[0].set_xlabel('Date', fontsize=12)
ax[0].set_ylabel('Total Cases', fontsize=12)
ax[0].tick_params(axis='x', rotation=45)
ax[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f'{int(x):,}'))

# Total Deaths
ax[1].plot(df_global['date'], df_global['total_deaths'], color='firebrick', linewidth=2)
ax[1].set_title('Global COVID-19 Deaths Over Time', fontsize=16)
ax[1].set_xlabel('Date', fontsize=12)
ax[1].set_ylabel('Total Deaths', fontsize=12)
ax[1].tick_params(axis='x', rotation=45)
ax[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f'{int(x):,}'))

plt.tight_layout()
plt.show()

# Calculate daily new cases and deaths for better visualization
df_global['rolling_new_cases'] = df_global['new_cases'].rolling(window=7).mean()
df_global['rolling_new_deaths'] = df_global['new_deaths'].rolling(window=7).mean()

# Plot new cases and deaths with 7-day rolling average
fig, ax = plt.subplots(1, 2, figsize=(20, 6))

# New Cases
ax[0].plot(df_global['date'], df_global['rolling_new_cases'], color='steelblue', linewidth=2)
ax[0].set_title('Global Daily New COVID-19 Cases (7-day average)', fontsize=16)
ax[0].set_xlabel('Date', fontsize=12)
ax[0].set_ylabel('New Cases', fontsize=12)
ax[0].tick_params(axis='x', rotation=45)
ax[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f'{int(x):,}'))

# New Deaths
ax[1].plot(df_global['date'], df_global['rolling_new_deaths'], color='firebrick', linewidth=2)
ax[1].set_title('Global Daily New COVID-19 Deaths (7-day average)', fontsize=16)
ax[1].set_xlabel('Date', fontsize=12)
ax[1].set_ylabel('New Deaths', fontsize=12)
ax[1].tick_params(axis='x', rotation=45)
ax[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f'{int(x):,}'))

plt.tight_layout()
plt.show()

# Filter out 'World' for country comparisons
df_countries_only = df_countries[df_countries['location'] != 'World'].copy()

# Get the latest data for each country
latest_data = df_countries_only.sort_values('date').groupby('location').last().reset_index()

# Sort countries by total cases
top_countries_by_cases = latest_data.sort_values('total_cases', ascending=False)

# Plot top countries by total cases
plt.figure(figsize=(12, 8))
sns.barplot(x='total_cases', y='location', data=top_countries_by_cases, palette='viridis')
plt.title('Total COVID-19 Cases by Country', fontsize=16)
plt.xlabel('Total Cases', fontsize=12)
plt.ylabel('Country', fontsize=12)
plt.xscale('log')  # Using log scale for better visualization
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.xticks(rotation=0)
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f'{int(x):,}'))
plt.tight_layout()
plt.show()

# Plot top countries by total deaths
top_countries_by_deaths = latest_data.sort_values('total_deaths', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='total_deaths', y='location', data=top_countries_by_deaths, palette='rocket')
plt.title('Total COVID-19 Deaths by Country', fontsize=16)
plt.xlabel('Total Deaths', fontsize=12)
plt.ylabel('Country', fontsize=12)
plt.xscale('log')  # Using log scale for better visualization
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.xticks(rotation=0)
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f'{int(x):,}'))
plt.tight_layout()
plt.show()

# Calculate death rate (deaths per case)
latest_data['death_rate'] = (latest_data['total_deaths'] / latest_data['total_cases']) * 100

# Plot death rate by country
plt.figure(figsize=(12, 8))
sns.barplot(x='death_rate', y='location', data=latest_data.sort_values('death_rate', ascending=False), palette='flare')
plt.title('COVID-19 Death Rate by Country (Deaths per 100 Cases)', fontsize=16)
plt.xlabel('Death Rate (%)', fontsize=12)
plt.ylabel('Country', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Plot total cases over time for selected countries
plt.figure(figsize=(15, 10))

# Get a list of countries excluding 'World'
countries = df_countries_only['location'].unique().tolist()

# Create a colormap
colors = plt.cm.viridis(np.linspace(0, 1, len(countries)))

for i, country in enumerate(countries):
    country_data = df_countries_only[df_countries_only['location'] == country]
    plt.plot(country_data['date'], country_data['total_cases'], label=country, linewidth=2, color=colors[i])

plt.title('COVID-19 Total Cases by Country', fontsize=18)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Total Cases (log scale)', fontsize=14)
plt.yscale('log')  # Using log scale to better visualize differences
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='upper left', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot new cases over time (7-day rolling average) for selected countries
plt.figure(figsize=(15, 10))

for i, country in enumerate(countries):
    country_data = df_countries_only[df_countries_only['location'] == country].copy()
    country_data['rolling_new_cases'] = country_data['new_cases'].rolling(window=7).mean()
    plt.plot(country_data['date'], country_data['rolling_new_cases'], label=country, linewidth=2, color=colors[i])

plt.title('COVID-19 Daily New Cases by Country (7-day average)', fontsize=18)
plt.xlabel('Date', fontsize=14)
plt.ylabel('New Cases (log scale)', fontsize=14)
plt.yscale('log')  # Using log scale to better visualize differences
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='upper left', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot cases per million for a fair comparison accounting for population
plt.figure(figsize=(15, 10))

for i, country in enumerate(countries):
    country_data = df_countries_only[df_countries_only['location'] == country]
    # Make sure we have the data column before plotting
    if 'total_cases_per_million' in country_data.columns and not country_data['total_cases_per_million'].isnull().all():
        plt.plot(country_data['date'], country_data['total_cases_per_million'], 
                 label=country, linewidth=2, color=colors[i])

plt.title('COVID-19 Cases per Million by Country', fontsize=18)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Cases per Million', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='upper left', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot vaccination progress for countries
plt.figure(figsize=(15, 10))

for i, country in enumerate(countries):
    country_data = df_countries_only[df_countries_only['location'] == country]
    # Make sure we have the data column before plotting
    if 'people_vaccinated_per_hundred' in country_data.columns and not country_data['people_vaccinated_per_hundred'].isnull().all():
        plt.plot(country_data['date'], country_data['people_vaccinated_per_hundred'], 
                 label=country, linewidth=2, color=colors[i])

plt.title('COVID-19 Vaccination Progress (% of Population with at least one dose)', fontsize=18)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Percentage of Population', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.ylim(0, 100)  # Set y-axis from 0-100%
plt.legend(loc='upper left', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Get the latest vaccination data for each country
latest_vax_data = latest_data.sort_values('people_fully_vaccinated_per_hundred', ascending=False)

# Plot vaccination coverage by country
plt.figure(figsize=(12, 8))
sns.barplot(x='people_fully_vaccinated_per_hundred', y='location', 
            data=latest_vax_data, palette='YlGnBu')
plt.title('COVID-19 Full Vaccination Coverage by Country', fontsize=16)
plt.xlabel('Percentage of Population Fully Vaccinated', fontsize=12)
plt.ylabel('Country', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.xlim(0, 100)  # Set x-axis from 0-100%
plt.tight_layout()
plt.show()

# Get the most recent data for each country in the full dataset
latest_global_data = df.sort_values('date').groupby('location').last().reset_index()

# Filter to keep only countries (remove continents and special regions)
latest_countries = latest_global_data[~latest_global_data['iso_code'].isnull() & 
                                     (latest_global_data['iso_code'].str.len() == 3)].copy()

# Create a choropleth map of total cases per million
fig = px.choropleth(
    latest_countries,
    locations="iso_code",
    color="total_cases_per_million",
    hover_name="location",
    hover_data=["total_cases", "total_deaths", "total_cases_per_million"],
    title="COVID-19 Total Cases per Million by Country",
    color_continuous_scale="Viridis",
    projection="natural earth"
)

fig.update_layout(
    coloraxis_colorbar=dict(
        title="Cases per Million"
    ),
    width=900,
    height=600
)

fig.show()

# Create a choropleth map of vaccination rates
fig = px.choropleth(
    latest_countries,
    locations="iso_code",
    color="people_fully_vaccinated_per_hundred",
    hover_name="location",
    hover_data=["people_fully_vaccinated_per_hundred", "people_vaccinated_per_hundred"],
    title="COVID-19 Vaccination Rate by Country (% Fully Vaccinated)",
    color_continuous_scale="YlGnBu",
    projection="natural earth"
)

fig.update_layout(
    coloraxis_colorbar=dict(
        title="% Fully Vaccinated"
    ),
    width=900,
    height=600
)

fig.show()

# Select relevant columns for correlation analysis
correlation_columns = [
    'total_cases_per_million', 
    'total_deaths_per_million',
    'people_fully_vaccinated_per_hundred',
    'aged_65_older',
    'gdp_per_capita',
    'human_development_index'
]

# Get correlation matrix
corr_data = latest_countries[correlation_columns].corr()

# Plot correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_data, annot=True, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
plt.title('Correlation Between COVID-19 Metrics and Country Characteristics', fontsize=16)
plt.tight_layout()
plt.show()

# Calculate some key statistics for our insights
global_latest = df_global.iloc[-1]
total_global_cases = global_latest['total_cases']
total_global_deaths = global_latest['total_deaths']
global_death_rate = (total_global_deaths / total_global_cases) * 100

# Get country with highest cases per million
highest_cases_per_million = latest_countries.sort_values('total_cases_per_million', ascending=False).iloc[0]
highest_cases_country = highest_cases_per_million['location']
highest_cases_value = highest_cases_per_million['total_cases_per_million']

# Get country with highest vaccination rate
highest_vax = latest_countries.sort_values('people_fully_vaccinated_per_hundred', ascending=False).iloc[0]
highest_vax_country = highest_vax['location']
highest_vax_rate = highest_vax['people_fully_vaccinated_per_hundred']

# Calculate average vaccination rate
avg_vax_rate = latest_countries['people_fully_vaccinated_per_hundred'].mean()

fig.update_layout(
    coloraxis_colorbar=dict(
        title="% Fully Vaccinated"
    ),
    width=900,
    height=600
)

fig.show()
