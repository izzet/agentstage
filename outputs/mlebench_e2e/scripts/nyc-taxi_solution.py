import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Load data - using a sample for faster training
print("Loading data...")
train = pd.read_csv('data/new-york-city-taxi-fare-prediction/labels.csv', 
                     nrows=500000)  # Use 500k rows for speed
test = pd.read_csv('data/new-york-city-taxi-fare-prediction/test.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# Extract features
def create_features(df):
    # Haversine distance calculation
    def haversine_distance(lat1, lon1, lat2, lon2):
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return 6371 * c  # Earth radius in km
    
    df = df.copy()
    
    # Extract time features
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df['hour'] = df['pickup_datetime'].dt.hour
    df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
    df['month'] = df['pickup_datetime'].dt.month
    df['day'] = df['pickup_datetime'].dt.day
    
    # Distance features
    df['distance'] = haversine_distance(
        df['pickup_latitude'].values,
        df['pickup_longitude'].values,
        df['dropoff_latitude'].values,
        df['dropoff_longitude'].values
    )
    
    # Location features
    df['pickup_longitude_lat_ratio'] = df['pickup_longitude'] / (df['pickup_latitude'] + 1e-6)
    df['dropoff_longitude_lat_ratio'] = df['dropoff_longitude'] / (df['dropoff_latitude'] + 1e-6)
    
    # Euclidean distance
    df['euclidean_distance'] = np.sqrt(
        (df['pickup_latitude'] - df['dropoff_latitude'])**2 +
        (df['pickup_longitude'] - df['dropoff_longitude'])**2
    )
    
    # Passenger count
    df['passenger_count'] = df['passenger_count'].fillna(1)
    
    return df

print("Creating features...")
train = create_features(train)
test = create_features(test)

# Select features for model
feature_cols = ['hour', 'day_of_week', 'month', 'day', 'passenger_count',
                'pickup_longitude', 'pickup_latitude', 
                'dropoff_longitude', 'dropoff_latitude',
                'distance', 'euclidean_distance',
                'pickup_longitude_lat_ratio', 'dropoff_longitude_lat_ratio']

# Filter out bad data (zero coordinates)
train = train[(train['pickup_longitude'] != 0) & (train['pickup_latitude'] != 0) &
              (train['dropoff_longitude'] != 0) & (train['dropoff_latitude'] != 0)]

# Filter outliers in fare
train = train[(train['fare_amount'] >= 2.5) & (train['fare_amount'] <= 200)]
train = train[(train['distance'] > 0) & (train['distance'] < 100)]

print(f"Train after filtering: {train.shape}")

# Prepare data
X_train = train[feature_cols].copy()
y_train = train['fare_amount'].copy()

X_test = test[feature_cols].copy()

# Handle any NaN or inf values
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

X_train = X_train.replace([np.inf, -np.inf], 0)
X_test = X_test.replace([np.inf, -np.inf], 0)

print("Training LightGBM model...")
# Train LightGBM
model = lgb.LGBMRegressor(
    n_estimators=30,
    num_leaves=31,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    verbose=-1,
    random_state=42
)

model.fit(X_train, y_train)

print("Making predictions...")
# Predict
y_pred = model.predict(X_test)

# Ensure predictions are reasonable
y_pred = np.clip(y_pred, 2.5, 200)

# Create submission
submission = pd.DataFrame({
    'key': test['key'],
    'fare_amount': y_pred
})

print(f"Submission shape: {submission.shape}")
print(f"Sample predictions:\n{submission.head()}")

submission.to_csv('submission.csv', index=False)
print("Submission saved to submission.csv")
