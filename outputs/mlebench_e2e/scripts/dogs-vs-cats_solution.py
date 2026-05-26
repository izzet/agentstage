import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Paths
train_path = 'data/dogs-vs-cats-redux-kernels-edition/train/'
test_path = 'data/dogs-vs-cats-redux-kernels-edition/test/'

print("Loading training data...")
X_train = []
y_train = []

# Process training images
train_files = sorted(os.listdir(train_path))
for fname in train_files:
    if fname.endswith('.jpg'):
        # Extract label from filename (cat.* = 0, dog.* = 1)
        label = 1 if fname.startswith('dog') else 0
        
        # Load and process image
        img_path = os.path.join(train_path, fname)
        try:
            img = Image.open(img_path).convert('RGB')
            # Resize to small size for speed
            img = img.resize((32, 32))
            # Convert to array and flatten
            img_array = np.array(img, dtype=np.float32).flatten()
            # Normalize
            img_array = img_array / 255.0
            
            X_train.append(img_array)
            y_train.append(label)
        except Exception as e:
            print(f"Error processing {fname}: {e}")

X_train = np.array(X_train)
y_train = np.array(y_train)

print(f"Loaded {len(X_train)} training images")
print(f"Class distribution: {np.sum(y_train)} dogs, {len(y_train) - np.sum(y_train)} cats")

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Train logistic regression (fast with small data)
print("Training model...")
model = LogisticRegression(max_iter=100, n_jobs=-1, random_state=42)
model.fit(X_train, y_train)

# Process test images
print("Processing test images...")
test_files = sorted(os.listdir(test_path))
test_ids = []
test_probs = []

X_test = []
for fname in test_files:
    if fname.endswith('.jpg'):
        # Extract ID from filename
        test_id = int(fname.replace('.jpg', ''))
        test_ids.append(test_id)
        
        # Load and process image
        img_path = os.path.join(test_path, fname)
        try:
            img = Image.open(img_path).convert('RGB')
            # Resize to same size as training
            img = img.resize((32, 32))
            # Convert to array and flatten
            img_array = np.array(img, dtype=np.float32).flatten()
            # Normalize
            img_array = img_array / 255.0
            
            X_test.append(img_array)
        except Exception as e:
            print(f"Error processing {fname}: {e}")
            X_test.append(np.zeros(32*32*3))

X_test = np.array(X_test)
X_test = scaler.transform(X_test)

# Make predictions
print("Making predictions...")
test_probs = model.predict_proba(X_test)[:, 1]  # Probability of class 1 (dog)

# Create submission
print("Creating submission...")
submission = pd.DataFrame({
    'id': test_ids,
    'label': test_probs
})

# Sort by ID to ensure correct order
submission = submission.sort_values('id').reset_index(drop=True)

# Save submission
submission.to_csv('/workspace/submission.csv', index=False)
print(f"Submission saved! Shape: {submission.shape}")
print(f"Sample predictions:\n{submission.head(10)}")
