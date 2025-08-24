import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, Union, List
from sklearn.linear_model import LogisticRegression

class DelayModel:

    def __init__(self):
        self._model = None

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        # Create a copy to avoid modifying original data
        processed_data = data.copy()
        
        # Create features used by the DS
        processed_data['period_day'] = processed_data['Fecha-I'].apply(self._get_period_day)
        processed_data['high_season'] = processed_data['Fecha-I'].apply(self._is_high_season)
        
        # Create min_diff and delay only if Fecha-O exists (for training)
        if 'Fecha-O' in processed_data.columns:
            processed_data['min_diff'] = processed_data.apply(self._get_min_diff, axis=1)
            processed_data['delay'] = np.where(processed_data['min_diff'] > 15, 1, 0)
        
        # Create one-hot encoded features for the top 10 features identified by DS
        features = pd.concat([
            pd.get_dummies(processed_data['OPERA'], prefix='OPERA'),
            pd.get_dummies(processed_data['TIPOVUELO'], prefix='TIPOVUELO'), 
            pd.get_dummies(processed_data['MES'], prefix='MES')], 
            axis=1
        )
        
        # Top 10 features identified from DS analysis
        top_10_features = [
            "OPERA_Latin American Wings", 
            "MES_7",
            "MES_10",
            "OPERA_Grupo LATAM",
            "MES_12",
            "TIPOVUELO_I",
            "MES_4",
            "MES_11",
            "OPERA_Sky Airline",
            "OPERA_Copa Air"
        ]
        
        # Ensure all features exist (fill missing with 0)
        for feature in top_10_features:
            if feature not in features.columns:
                features[feature] = 0
                
        # Select only the top 10 features
        features = features[top_10_features]
        
        if target_column and target_column in processed_data.columns:
            target = processed_data[[target_column]]
            return features, target
        else:
            return features

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        # Calculate class weights for balancing (based on DS analysis)
        n_y0 = len(target[target.iloc[:, 0] == 0])
        n_y1 = len(target[target.iloc[:, 0] == 1])
        
        # Use LogisticRegression with class balancing as chosen by DS analysis
        self._model = LogisticRegression(
            class_weight={1: n_y0/len(target), 0: n_y1/len(target)}
        )
        
        self._model.fit(features, target.values.ravel())

    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            (List[int]): predicted targets.
        """
        if self._model is None:
            # Auto-train the model on first prediction using the full dataset
            data = pd.read_csv("data/data.csv", low_memory=False)
            train_features, target = self.preprocess(data, target_column="delay")
            self.fit(train_features, target)
            
        predictions = self._model.predict(features)
        return [int(pred) for pred in predictions]

    def _get_period_day(self, date):
        """Get period of day from date string"""
        date_time = datetime.strptime(date, '%Y-%m-%d %H:%M:%S').time()
        morning_min = datetime.strptime("05:00", '%H:%M').time()
        morning_max = datetime.strptime("11:59", '%H:%M').time()
        afternoon_min = datetime.strptime("12:00", '%H:%M').time()
        afternoon_max = datetime.strptime("18:59", '%H:%M').time()
        evening_min = datetime.strptime("19:00", '%H:%M').time()
        evening_max = datetime.strptime("23:59", '%H:%M').time()
        night_min = datetime.strptime("00:00", '%H:%M').time()
        night_max = datetime.strptime("4:59", '%H:%M').time()
        
        if morning_min <= date_time <= morning_max:
            return 'mañana'
        elif afternoon_min <= date_time <= afternoon_max:
            return 'tarde'
        elif (evening_min <= date_time <= evening_max) or (night_min <= date_time <= night_max):
            return 'noche'

    def _is_high_season(self, fecha):
        """Check if date is in high season"""
        fecha_año = int(fecha.split('-')[0])
        fecha = datetime.strptime(fecha, '%Y-%m-%d %H:%M:%S')
        range1_min = datetime.strptime(f'15-Dec-{fecha_año}', '%d-%b-%Y')
        range1_max = datetime.strptime(f'31-Dec-{fecha_año}', '%d-%b-%Y')
        range2_min = datetime.strptime(f'1-Jan-{fecha_año}', '%d-%b-%Y')
        range2_max = datetime.strptime(f'3-Mar-{fecha_año}', '%d-%b-%Y')
        range3_min = datetime.strptime(f'15-Jul-{fecha_año}', '%d-%b-%Y')
        range3_max = datetime.strptime(f'31-Jul-{fecha_año}', '%d-%b-%Y')
        range4_min = datetime.strptime(f'11-Sep-{fecha_año}', '%d-%b-%Y')
        range4_max = datetime.strptime(f'30-Sep-{fecha_año}', '%d-%b-%Y')
        
        if ((fecha >= range1_min and fecha <= range1_max) or 
            (fecha >= range2_min and fecha <= range2_max) or 
            (fecha >= range3_min and fecha <= range3_max) or
            (fecha >= range4_min and fecha <= range4_max)):
            return 1
        else:
            return 0

    def _get_min_diff(self, data):
        """Calculate minute difference between actual and scheduled time"""
        fecha_o = datetime.strptime(data['Fecha-O'], '%Y-%m-%d %H:%M:%S')
        fecha_i = datetime.strptime(data['Fecha-I'], '%Y-%m-%d %H:%M:%S')
        min_diff = ((fecha_o - fecha_i).total_seconds())/60
        return min_diff