import pandas as pd
import sns as sns
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from gdrive_downloader import g_downloader
import gdown

import utils


class DataPreprocess:
    def __init__(self, file_path: str = 'train.csv'):
        """
        Initialize the DataPreprocessing object.

        Args:
        - file_path (str): The path to the CSV file.
        """
        print('Downloading data')
        g_downloader('1IB1lMoRYQ3g--cFmGx6UEvmLN3YSyWUP')
        print('Data Downloaded')
        self.preprocessed_df = None
        self.df_categorical = None
        self.df_cont = None
        self.file_path = file_path
        self.df = None

    def load_dataset(self):
        """
        Load the dataset from the specified CSV file.
        """
        self.df = pd.read_csv(self.file_path)
        return self.df

    def data_quality_report_continuous_variables(self):
        """
        Generate a data quality report for continuous variables.
        """
        self.df_cont = self.df.drop(
            columns=['EntryStreetName', 'ExitStreetName', 'EntryHeading', 'ExitHeading', 'Path', 'City'])
        data_types = pd.DataFrame(self.df_cont.dtypes, columns=['Data Type'])
        total_count = pd.DataFrame(self.df_cont.count(), columns=['Count'])
        percentage_missing = pd.DataFrame(self.df_cont.isnull().sum() / len(self.df) * 100, columns=['%Miss'])

        cardinality = pd.DataFrame(columns=['Cardinality'])
        for v in list(self.df_cont.columns.values):
            cardinality.loc[v] = [self.df_cont[v].nunique()]

        minimum = pd.DataFrame(columns=['Min'])
        for v in list(self.df_cont.columns.values):
            minimum.loc[v] = [self.df_cont[v].min()]

        first_quartile = pd.DataFrame(columns=['1st Qrt'])
        for v in list(self.df_cont.columns.values):
            first_quartile.loc[v] = [self.df_cont[v].quantile(0.25)]

        mean = pd.DataFrame(columns=['Mean'])
        for v in list(self.df_cont.columns.values):
            mean.loc[v] = [self.df_cont[v].mean()]

        median = pd.DataFrame(columns=['Median'])
        for v in list(self.df_cont.columns.values):
            median.loc[v] = [self.df_cont[v].median()]

        third_quartile = pd.DataFrame(columns=['3rd Qrt'])
        for v in list(self.df_cont.columns.values):
            third_quartile.loc[v] = [self.df_cont[v].quantile(0.75)]

        maximum = pd.DataFrame(columns=['Max'])
        for v in list(self.df_cont.columns.values):
            maximum.loc[v] = [self.df_cont[v].max()]

        stddev = pd.DataFrame(columns=['Std_Dev'])
        for v in list(self.df_cont.columns.values):
            stddev.loc[v] = [self.df_cont[v].std()]
            data_quality_report = data_types.join(total_count).join(percentage_missing).join(cardinality).join(
                minimum).join(first_quartile).join(mean).join(median).join(third_quartile).join(maximum).join(stddev)
            print("\nData Quality Report for continuous variables")
            print("Total records: {}".format(len(data_quality_report.index)))
            print(data_quality_report.round(2))
            return data_quality_report.round(2)

    def generate_categorical_data_quality_report(self):
        """
        Generate a data quality report for categorical variables.
        """
        self.df_categorical = self.df[
            ['EntryStreetName', 'ExitStreetName', 'EntryHeading', 'ExitHeading', 'Path', 'City']].copy()
        mode = pd.DataFrame(columns=['Mode'])
        for v in list(self.df_categorical.columns.values):
            mode.loc[v] = [self.df_categorical[v].value_counts().index.tolist()[0]]
            mode.loc['EntryStreetName'] = [self.df_categorical['EntryStreetName'].value_counts().index.tolist()[0]]

        data_types = pd.DataFrame(self.df_categorical.dtypes, columns=['Data Type'])
        total_count = pd.DataFrame(self.df_categorical.count(), columns=['Count'])
        percentage_missing = pd.DataFrame(self.df_categorical.isnull().sum() / len(self.df) * 100, columns=['%Miss'])

        cardinality = pd.DataFrame(columns=['Cardinality'])
        for v in list(self.df_categorical.columns.values):
            cardinality.loc[v] = [self.df_categorical[v].nunique()]

        mode = pd.DataFrame(columns=['Mode'])
        for v in list(self.df_categorical.columns.values):
            mode.loc[v] = [self.df_categorical[v].value_counts().index.tolist()[0]]

        mode_freq = pd.DataFrame(columns=['Mode Freq'])
        for v in list(self.df_categorical.columns.values):
            mode_freq.loc[v] = [self.df_categorical[v].value_counts()[0]]

        mode_perc = pd.DataFrame(columns=['Mode Perc'])
        for v in list(self.df_categorical.columns.values):
            mode_perc.loc[v] = [
                self.df_categorical[v].value_counts()[0] / len(self.df_categorical['EntryStreetName']) * 100]

        smode = pd.DataFrame(columns=['Second Mode'])
        for v in list(self.df_categorical.columns.values):
            smode.loc[v] = [self.df_categorical[v].value_counts().index.tolist()[1]]

        smode_freq = pd.DataFrame(columns=['Second Mode Freq'])
        for v in list(self.df_categorical.columns.values):
            smode_freq.loc[v] = [self.df_categorical[v].value_counts()[1]]

        smode_perc = pd.DataFrame(columns=['Second Mode Perc'])
        for v in list(self.df_categorical.columns.values):
            smode_perc.loc[v] = [
                self.df_categorical[v].value_counts()[1] / len(self.df_categorical['EntryStreetName']) * 100]

        data_quality_report2 = data_types.join(total_count).join(percentage_missing).join(cardinality).join(mode).join(
            mode_freq).join(mode_perc).join(smode).join(smode_freq).join(smode_perc)
        print("\n Data Quality Report - Categorical Features")
        print("Total records: {}".format(len(data_quality_report2.index)))
        print(data_quality_report2)
        return data_quality_report2

    def data_preprocessing(self):
        """
        Performs data preprocessing on the input dataframe.

        Returns:
            X_train (DataFrame): Feature matrix for training.
            y_train (DataFrame): Target variables for training.
            X_test (DataFrame): Feature matrix for testing.
            y_test (DataFrame): Target variables for testing.
        """
        # Map directions to numeric values
        directions = {
            'N': 0,
            'NE': 1 / 4,
            'E': 1 / 2,
            'SE': 3 / 4,
            'S': 1,
            'SW': 5 / 4,
            'W': 3 / 2,
            'NW': 7 / 4
        }
        self.df['EntryHeading'] = self.df['EntryHeading'].map(directions)
        self.df['ExitHeading'] = self.df['ExitHeading'].map(directions)

        road_encoding = {
            'Road': 1,
            'Street': 2,
            'Avenue': 2,
            'Drive': 3,
            'Broad': 3,
            'Boulevard': 4
        }

        def encode(x):
            if pd.isna(x):
                return 0
            for road in road_encoding.keys():
                if road in x:
                    return road_encoding[road]

            return 0

        self.df['EntryTypeStreet'] = self.df['EntryStreetName'].apply(encode)
        self.df['ExitTypeStreet'] = self.df['ExitStreetName'].apply(encode)

        # Calculate difference in heading
        self.df['diffHeading'] = self.df['EntryHeading'] - self.df['ExitHeading']

        # Create a binary column indicating if the entry and exit street names are the same
        self.df["same_street_exact"] = (self.df["EntryStreetName"] == self.df["ExitStreetName"]).astype(int)

        # Label encode the intersection column
        le = preprocessing.LabelEncoder()
        self.df["Intersection"] = self.df["IntersectionId"].astype(str) + self.df["City"]
        le.fit(pd.concat([self.df["Intersection"]]).drop_duplicates().values)
        self.df["Intersection"] = le.transform(self.df["Intersection"])

        """Adding temperature (°F) of each city by month"""
        # Reference: https://www.kaggle.com/dcaichara/feature-engineering-and-lightgbm
        monthly_avg = {'Atlanta1': 43.0, 'Atlanta5': 68.5, 'Atlanta6': 76.0, 'Atlanta7': 78.0, 'Atlanta8': 78.0,
                       'Atlanta9': 72.5,
                       'Atlanta10': 62.0, 'Atlanta11': 52.5, 'Atlanta12': 45.0, 'Boston1': 29.5, 'Boston5': 58.5,
                       'Boston6': 68.0,
                       'Boston7': 74.0, 'Boston8': 73.0, 'Boston9': 65.5, 'Boston10': 54.5, 'Boston11': 45.0,
                       'Boston12': 35.0,
                       'Chicago1': 27.0, 'Chicago5': 59.5, 'Chicago6': 70.0, 'Chicago7': 76.0, 'Chicago8': 75.5,
                       'Chicago9': 68.0,
                       'Chicago10': 56.0, 'Chicago11': 44.5, 'Chicago12': 32.0, 'Philadelphia1': 34.5,
                       'Philadelphia5': 66.0,
                       'Philadelphia6': 75.5, 'Philadelphia7': 80.5, 'Philadelphia8': 78.5, 'Philadelphia9': 71.5,
                       'Philadelphia10': 59.5,
                       'Philadelphia11': 49.0, 'Philadelphia12': 40.0}
        # Concatenating the city and month into one variable
        self.df['city_month'] = self.df["City"] + self.df["Month"].astype(str)
        # Creating a new column by mapping the city_month variable to it's corresponding average monthly temperature
        self.df["average_temp"] = self.df['city_month'].map(monthly_avg)

        """Adding rainfall (inches) of each city by month"""
        monthly_rainfall = {'Atlanta1': 5.02, 'Atlanta5': 3.95, 'Atlanta6': 3.63, 'Atlanta7': 5.12, 'Atlanta8': 3.67,
                            'Atlanta9': 4.09,
                            'Atlanta10': 3.11, 'Atlanta11': 4.10, 'Atlanta12': 3.82, 'Boston1': 3.92, 'Boston5': 3.24,
                            'Boston6': 3.22,
                            'Boston7': 3.06, 'Boston8': 3.37, 'Boston9': 3.47, 'Boston10': 3.79, 'Boston11': 3.98,
                            'Boston12': 3.73,
                            'Chicago1': 1.75, 'Chicago5': 3.38, 'Chicago6': 3.63, 'Chicago7': 3.51, 'Chicago8': 4.62,
                            'Chicago9': 3.27,
                            'Chicago10': 2.71, 'Chicago11': 3.01, 'Chicago12': 2.43, 'Philadelphia1': 3.52,
                            'Philadelphia5': 3.88,
                            'Philadelphia6': 3.29, 'Philadelphia7': 4.39, 'Philadelphia8': 3.82, 'Philadelphia9': 3.88,
                            'Philadelphia10': 2.75,
                            'Philadelphia11': 3.16, 'Philadelphia12': 3.31}

        # Creating a new column by mapping the city_month variable to it's corresponding average monthly rainfall
        self.df["average_rainfall"] = self.df['city_month'].map(monthly_rainfall)

        # standardization
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        lat_long = ['Latitude', 'Longitude']
        for col in lat_long:
            self.df[col] = (scaler.fit_transform(self.df[col].values.reshape(-1, 1)))

        # Create dummy variables for the city column
        self.preprocessed_df = pd.concat([self.df, pd.get_dummies(self.df["City"], dummy_na=False, drop_first=False)],
                                         axis=1).drop(["City"], axis=1)

        # Define the feature columns
        FEAT_COLS = ['Intersection',
                     #  "IntersectionId",
                     'diffHeading',
                     'same_street_exact',
                     "Hour",
                     "Weekend",
                     "Month",
                     'Latitude',
                     'Longitude',
                     'EntryHeading',
                     'ExitHeading',
                     'EntryTypeStreet',
                     'ExitTypeStreet',
                     'average_temp',
                     'average_rainfall',
                     'Atlanta',
                     'Boston',
                     'Chicago',
                     'Philadelphia']

        # Split the data into train, validation, and test sets with 80-10-10 ratio
        train_df, test_df = train_test_split(self.preprocessed_df, test_size=0.2, random_state=42)

        # Print the number of samples in each set
        print(f'Train set size: {len(train_df)}')
        print(f'Test set size: {len(test_df)}')

        # Prepare the feature matrix and target variables for training
        X_train = train_df[FEAT_COLS]
        y_train = train_df[
            ['TotalTimeStopped_p20', 'TotalTimeStopped_p50', 'TotalTimeStopped_p80', 'DistanceToFirstStop_p20',
             'DistanceToFirstStop_p50', 'DistanceToFirstStop_p80']]

        # test set
        X_test = test_df[FEAT_COLS]
        y_test = test_df[
            ['TotalTimeStopped_p20', 'TotalTimeStopped_p50', 'TotalTimeStopped_p80', 'DistanceToFirstStop_p20',
             'DistanceToFirstStop_p50', 'DistanceToFirstStop_p80']]
        return X_train, y_train, X_test, y_test
