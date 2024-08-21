import os
import time
import numpy as np
import pandas as pd

"Code credits (most of it): FIlip Radovic from Universuty of Tübingen"

class GermanWeatherEnergyData:
    """Class to load and preprocess the German weather and
    energy data for the short-term (1h ahead) and long-term (24 hrs ahead) models.
    """
    def __init__(self,
                 target_idx=62,
                 window_size=24,  # 24 hours
                 short_horizon=1,  # 1 hour
                 long_horizon=24,  # 24 hours
                 data_dir=None):
        """Arguments:
        - target_idx: index of the target column in the dataset
        - window_size: number of hours to consider in the past for the prediction
        - short_horizon: number of hours ahead to predict in the short-term model
        - long_horizon: number of hours ahead to predict in the long-term model
        - data_dir: directory containing the data files
        """
        ## Load and preprocess data
        print('Loading and preprocessing data...')
        self.data_dir = data_dir
        self.target_idx = target_idx
        self.window_size = window_size
        self.short_horizon = short_horizon
        self.long_horizon = long_horizon
        self.generate_dataset()
        print('Processed data generated successfully.')

    def get_train_val_test_data(self):
        # Split the data by year
        train_data = self.trainval_data[(self.trainval_data['time'].dt.year == 2019) | (self.trainval_data['time'].dt.year == 2020)]
        val_data = self.trainval_data[self.trainval_data['time'].dt.year == 2021]
        test_data = self.test_data[self.test_data['time'].dt.year == 2022]

        feature_columns = list(train_data.columns)
        # feature_columns.remove(train_data.columns[self.target_idx])
        print("From function get_train_val_test_data:")
        print(train_data.columns[self.target_idx])
        X_train = train_data[feature_columns]
        y_train = train_data.iloc[:, self.target_idx]
        X_val = val_data[feature_columns]
        y_val = val_data.iloc[:, self.target_idx]
        X_test = test_data[feature_columns]
        y_test = test_data.iloc[:, self.target_idx]
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
        
    
    def prepare_data(self, X, y):
        "Prepare the data for training the models."
        # data = data.set_index('time').sort_index()
        
        X = X.to_numpy()
        y = y.to_numpy()

        num_samples = len(X) - self.window_size - max(self.short_horizon, self.long_horizon) + 1

        # Prepare inputs and targets for both short and long horizon models
        X_flattened = np.array([X[i:i + self.window_size].flatten() for i in range(num_samples)])
        y_short = np.array([y[i + self.window_size + self.short_horizon - 1] for i in range(num_samples)])
        y_long = np.array([y[i + self.window_size + self.long_horizon - 1] for i in range(num_samples)])
        
        return (X_flattened, y_short), (X_flattened, y_long) 
      
        

    def generate_dataset(self):
        def fill_missing_weather_data(weather_data):
            previous_start = weather_data['time'].min()
            weather_data['time'] = weather_data['time'] - pd.Timedelta(days=1)
            weather_data = weather_data[weather_data['time'] >= previous_start]
            return weather_data

        def fill_missing_prices_data(prices_data):
            prices_data.set_index('time', inplace=True)
            for column in prices_data.columns:
                if prices_data[column].dtype == 'object':
                    prices_data[column] = prices_data[column].str.replace(',', '.')
                    prices_data[column] = pd.to_numeric(prices_data[column], errors='coerce')
            # prices_data = prices_data.resample('1H').interpolate(method='linear')
            return prices_data.reset_index()

        def fill_missing_realcap_data(realcap_data):
            realcap_data.set_index('time', inplace=True)
            realcap_data = realcap_data[~realcap_data.index.duplicated(keep='first')]
            for column in realcap_data.columns:
                if realcap_data[column].dtype == 'object':
                    realcap_data[column] = realcap_data[column].str.replace('.', '').str.replace(',', '.')
                    realcap_data[column] = pd.to_numeric(realcap_data[column], errors='coerce')
            start_date = '2018-12-31 23:00:00'
            end_date = '2022-12-31 22:45:00'
            new_index = pd.date_range(start=start_date, end=end_date, freq='15min')
            realcap_data = realcap_data.reindex(new_index).ffill()
            return realcap_data .resample('1h').mean().reset_index().rename(columns={'index': 'time'})


        def convert_to_UTC(data, source='Europe/Berlin'):
            data['time'] = data['time'].dt.tz_localize('Europe/Berlin', ambiguous='NaT').dt.tz_convert('UTC')
            data['time'] = data['time'].dt.tz_localize(None)
            data['time'] = data['time'].interpolate()
            return data
        
        def split_time_into_integers(data):
            data['year'] = data['time'].dt.year
            data['month_cos'] = np.round(np.cos(data['time'].dt.month / 12 * 2 * np.pi), 2)
            data['month_sin'] = np.round(np.sin(data['time'].dt.month / 12 * 2 * np.pi), 2)
            data['day_cos'] = np.round(np.cos(data['time'].dt.day / 31 * 2 * np.pi), 2)
            data['day_sin'] = np.round(np.sin(data['time'].dt.day / 31 * 2 * np.pi), 2)
            data['hour_cos'] = np.round(np.cos(data['time'].dt.hour / 24 * 2 * np.pi), 2)
            data['hour_sin'] = np.round(np.sin(data['time'].dt.hour / 24 * 2 * np.pi), 2)
            # data.drop(columns=['time'], inplace=True)
            return data

        # Load data
        start_time = time.time()
        weather_data19 = pd.read_csv(os.path.join(self.data_dir, "weather_data_19-21_de.csv")).drop(columns=['forecast_origin']) # to be used as training/validation data
        weather_data22 = pd.read_csv(os.path.join(self.data_dir, "weather_data_22_de.csv")).drop(columns=['forecast_origin']) # to be used as test data
        print(f"Loaded weather data in {time.time() - start_time:.2f} seconds.")

        ## Load the energy data
        start_time = time.time()
        # Prices data of length (365 * 4 + 1) * 24 = 35064
        prices_data = pd.read_csv(os.path.join(self.data_dir, "energy_data/prices_eu.csv"), delimiter=';')
        prices_data = prices_data[['Date from', 'Germany/Luxembourg [€/MWh]']].rename(columns={'Date from': 'time', 'Germany/Luxembourg [€/MWh]': 'price'})
        # Capacities data of length 4
        installed_capacity_data = pd.read_csv(os.path.join(self.data_dir, 'energy_data/installed_capacity_de.csv'), delimiter=';').rename(columns={'Date from': 'time'})
        installed_capacity_data = installed_capacity_data[['time', 'Wind Offshore [MW] ', 'Wind Onshore [MW]', 'Photovoltaic [MW]']]
        installed_capacity_data.rename(columns={'Wind Offshore [MW] ': 'wind_offshore_capacity', 
                                                'Wind Onshore [MW]': 'wind_onshore_capacity', 
                                                'Photovoltaic [MW]': 'photovoltaic_capacity'}, inplace=True)
        # Realised supply of length (365 * 4 + 1) * 24 * 4 = 140256
        realised_supply_data = pd.read_csv(os.path.join(self.data_dir, 'energy_data/realised_supply_de.csv'), delimiter=';').rename(columns={'Date from': 'time'})
        realised_supply_data = realised_supply_data[['time', 'Wind Offshore [MW] ', 'Wind Onshore [MW]', 'Photovoltaic [MW]']]
        realised_supply_data.rename(columns={'Wind Offshore [MW] ': 'wind_offshore_supply',
                                            'Wind Onshore [MW]': 'wind_onshore_supply',
                                            'Photovoltaic [MW]': 'photovoltaic_supply'}, inplace=True)
        # Realised demand of length (365 * 4 + 1) * 24 * 4 = 140256
        realised_demand_data = pd.read_csv(os.path.join(self.data_dir, 'energy_data/realised_demand_de.csv'), delimiter=';').rename(columns={'Date from': 'time'})
        print(f"Loaded energy data in {time.time() - start_time:.2f} seconds.")

        ## Merge data
        start_time = time.time()
        realised_demand_data = realised_demand_data.drop(columns=['time', 'Date to'])
        realisation_data = pd.merge(realised_supply_data, realised_demand_data, left_index=True, right_index=True)
        print(f"Merged realisation data in {time.time() - start_time:.2f} seconds.")

        # Convert to datetime format
        start_time = time.time()
        weather_data19['time'] = pd.to_datetime(weather_data19['time'], format='%Y-%m-%d %H:%M:%S')
        weather_data22['time'] = pd.to_datetime(weather_data22['time'], format='%Y-%m-%d %H:%M:%S')
        realisation_data['time'] = pd.to_datetime(realisation_data['time'], format='%d.%m.%y %H:%M')
        installed_capacity_data['time'] = pd.to_datetime(installed_capacity_data['time'], format='%d.%m.%y')
        prices_data['time'] = pd.to_datetime(prices_data['time'], format='%d.%m.%y %H:%M')
        print(f"Converted to datetime format in {time.time() - start_time:.2f} seconds.")
        
        # Convert energy data to UTC
        start_time = time.time()
        prices_data = convert_to_UTC(prices_data)
        realisation_data = convert_to_UTC(realisation_data)
        installed_capacity_data = convert_to_UTC(installed_capacity_data)
        print(f"Converted energy data to UTC in {time.time() - start_time:.2f} seconds.")

        start_time = time.time()
        weather_data19 = fill_missing_weather_data(weather_data19)
        weather_data22 = fill_missing_weather_data(weather_data22)
        print(f"Filled missing weather data in {time.time() - start_time:.2f} seconds.")

        
        start_time = time.time()
        realisation_data = fill_missing_realcap_data(realisation_data)
        prices_data = fill_missing_prices_data(prices_data)
        installed_capacity_data = fill_missing_realcap_data(installed_capacity_data)

        # Perform the quadrant-based clustering
        start_time = time.time()

        def categorize_location(row, mid_latitude, mid_longitude):
            if row['latitude'] >= mid_latitude and row['longitude'] <= mid_longitude:
                return 'top_left'
            elif row['latitude'] >= mid_latitude and row['longitude'] > mid_longitude:
                return 'top_right'
            elif row['latitude'] < mid_latitude and row['longitude'] <= mid_longitude:
                return 'bottom_left'
            else:
                return 'bottom_right'

        def categorize_and_aggregate(weather_data):
            mid_latitude = weather_data['latitude'].mean()
            mid_longitude = weather_data['longitude'].mean()
            weather_data['region'] = weather_data.apply(lambda row: categorize_location(row, mid_latitude, mid_longitude), axis=1)
            weather_data.drop(columns=['longitude', 'latitude'], inplace=True)
            
            data_grouped = weather_data.groupby(['time', 'region']).mean().reset_index()
            data_pivoted = data_grouped.pivot(index='time', columns='region')
            data_pivoted.columns = [f'{col[1]}_{col[0]}' for col in data_pivoted.columns]
            data_pivoted.reset_index(inplace=True)
            return data_pivoted

        weather_data19 = categorize_and_aggregate(weather_data19)
        weather_data22 = categorize_and_aggregate(weather_data22)

        print(f"Clustered and aggregated locations in {time.time() - start_time:.2f} seconds.")




        energy_data = pd.merge(pd.merge(realisation_data, installed_capacity_data, on='time'), prices_data, how='left', on='time')
        energy_data['price'] = energy_data['price'].interpolate()
        # energy_data = aggregate_to_hourly(energy_data)
        print(f"Filled missing energy data in {time.time() - start_time:.2f} seconds.")

        start_time = time.time()
        self.trainval_data = pd.merge(weather_data19, energy_data, how='left', on='time')
        self.test_data = pd.merge(weather_data22, energy_data, how='left', on='time').ffill()
        print(f"Combined weather and energy data in {time.time() - start_time:.2f} seconds.")

        start_time = time.time()
        self.trainval_data = split_time_into_integers(self.trainval_data)
        self.test_data = split_time_into_integers(self.test_data)
        print(f"Split time into integers in {time.time() - start_time:.2f} seconds.")