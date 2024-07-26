### Exploring Credit Card Fraud Detection with Multiple Feature Selection and Optimization Techniques

Here's an in-depth implementation of various feature selection methods, combined with autoencoders and different classifiers to detect credit card fraud. We also employ different optimization techniques like PSO (Particle Swarm Optimization) to fine-tune the model's parameters. The aim is to create a comprehensive pipeline that includes data preprocessing, feature selection, model training, and evaluation.

### Setup and Initial Exploration

```python
# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Mount Google Drive (specific to Colab)
from google.colab import drive
drive.mount('/content/gdrive')

# Load the dataset
data = pd.read_csv('/content/gdrive/MyDrive/Swarm/CreditcardfraudDatacsvDroppped.csv', sep=',')
data.head()
```

### Data Inspection and Preparation

```python
# Check for missing values and data shape
data.shape
data.isnull().sum()
data['is_fraud'].value_counts()
data.info()

# Descriptive statistics
normal_df = data[data['is_fraud'] == 0]
fraud_df = data[data['is_fraud'] == 1]
normal_df.describe()
fraud_df.describe()
```

### Exploratory Data Analysis (EDA)

```python
import numpy as np
import matplotlib.pyplot as plt

# Plot distribution of transaction amounts for normal and fraud transactions
bins = np.linspace(200, 1000, 100)
plt.figure(figsize=(14, 8))
plt.hist(normal_df['amt'], bins, alpha=1, density=True, label='Normal')
plt.hist(fraud_df['amt'], bins, alpha=0.6, density=True, label='Fraud')
plt.legend(loc='upper right')
plt.title("Amount by percentage of transactions (transactions $200+)")
plt.xlabel("Transaction amount (USD)")
plt.ylabel("Percentage of transactions (%)")
plt.show()
```

### Feature Engineering and Selection

#### Label Encoding

```python
# Separate features and target variable
features = data.drop('is_fraud', axis=1)
target = data['is_fraud']

# Function to encode categorical features
def encode_categorical(df):
    categorical_cols = [col for col in df.columns if df[col].dtype == 'object']
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    return df

features = encode_categorical(features.copy())
print(features.head())
```

#### Chi-Square Test

```python
from sklearn.feature_selection import chi2

# Calculate chi-square scores and p-values
chi2_scores, p_values = chi2(features, target)

# Print results
results = pd.DataFrame({'feature': features.columns, 'chi2_score': chi2_scores, 'p_value': p_values})
print(results.sort_values(by=['p_value'], ascending=True))

# Select significant features
p_val_threshold = 0.05
significant_features = results[results['p_value'] <= p_val_threshold]['feature'].tolist()
print("Significant Features:", significant_features)
```

#### Correlation Analysis

```python
# Calculate correlation coefficients
correlations = features.corrwith(target)
correlation_matrix = pd.DataFrame(correlations, columns=['Correlation'])
correlation_matrix.index.name = 'Feature'
print(correlation_matrix)

# Select features with strong correlations
corr_threshold = 0.3
important_features = correlations[abs(correlations) >= corr_threshold].index
print("Important Features:", important_features.tolist())
```

#### Recursive Feature Elimination (RFE)

```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# RFE for feature selection
selector = RFE(estimator=LogisticRegression(), n_features_to_select=10, step=1)
selector.fit(features, target)
selected_features_rfe = features.columns[selector.support_]
print("Selected Features using RFE:", selected_features_rfe)
```

#### LASSO Regression

```python
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel

# Standardize features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# LASSO model for feature selection
lasso = Lasso(alpha=0.01)
selector = SelectFromModel(lasso)
selector.fit(features_scaled, target)
selected_features_lasso = features.columns[selector.get_support()]
print("Selected Features using LASSO:", selected_features_lasso)
```

### Model Training and Evaluation

#### Autoencoder for Feature Extraction

```python
# Feature scaling
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Split data into training and testing sets
train_x, test_x, train_y, test_y = train_test_split(features_scaled, target, test_size=0.2, random_state=42, stratify=target)

# Autoencoder model
autoencoder = Sequential([
    Dense(256, activation='relu', input_shape=(train_x.shape[1],)),
    Dense(128, activation='relu'),  # Bottleneck layer
    Dense(256, activation='relu'),
    Dense(train_x.shape[1], activation='sigmoid')
])

autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy')
autoencoder.fit(train_x, train_x, epochs=20, batch_size=32, validation_data=(test_x, test_x))

# Extract features from bottleneck layer
encoder = Sequential([
    autoencoder.layers[0],  # Encoder layer 1
    autoencoder.layers[1]   # Encoder layer 2
])

encoded_train = encoder.predict(train_x)
encoded_test = encoder.predict(test_x)

# Train Random Forest Classifier using encoded features
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(encoded_train, train_y)
predictions = rf_model.predict(encoded_test)
print(classification_report(test_y, predictions))
print("Confusion Matrix:", confusion_matrix(test_y, predictions))
```

### Optimization with Particle Swarm Optimization (PSO)

```python
from pyswarm import pso
from sklearn.metrics import f1_score

# Define objective function for PSO
def rf_objective(params):
    n_estimators, max_depth = int(params[0]), int(params[1])
    rf_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    rf_model.fit(encoded_train, train_y)
    predictions = rf_model.predict(encoded_test)
    score = f1_score(test_y, predictions)
    return -score  # Minimize negative f1-score

# PSO bounds for n_estimators and max_depth
lb = [10, 1]
ub = [200, 20]
best_params, _ = pso(rf_objective, lb, ub, swarmsize=20, maxiter=30)

# Train and evaluate model with optimized parameters
n_estimators_opt, max_depth_opt = int(best_params[0]), int(best_params[1])
rf_model = RandomForestClassifier(n_estimators=n_estimators_opt, max_depth=max_depth_opt, random_state=42)
rf_model.fit(encoded_train, train_y)
predictions = rf_model.predict(encoded_test)
print(classification_report(test_y, predictions))
print("Confusion Matrix:", confusion_matrix(test_y, predictions))
```

### Comparison with Deep Neural Networks (DNNs)

```python
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def credit_card_fraud_detection_with_dnn(data_path):
    # Load and preprocess data
    data = pd.read_csv(data_path, sep=',')
    X = data.drop('is_fraud', axis=1)
    y = data['is_fraud']

    # Label encode categorical features
    le = LabelEncoder()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_features:
        X[col] = le.fit_transform(X[col])

    # Feature scaling and PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=0.95)
    X_reduced = pca.fit_transform(X_scaled)

    # Split data and apply SMOTE
    X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42, stratify=y)
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    # Define DNN model
    def create_dnn(input_dim):
        model = Sequential([
            Dense(128, activation='relu', input_dim=input_dim),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        return model

    input_dim = X_resampled.shape[1]
    dnn = create_dnn(input_dim)
    dnn.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    # Early

 stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train model
    dnn.fit(X_resampled, y_resampled, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

    # Evaluate model
    y_pred = (dnn.predict(X_test) > 0.5).astype("int32")
    accuracy = accuracy_score(y_test, y_pred)
    print("DNN Model Accuracy:", accuracy)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

# Call the function with your dataset
data_path = '/content/gdrive/MyDrive/Swarm/CreditcardfraudDatacsvDroppped.csv'
credit_card_fraud_detection_with_dnn(data_path)
```

### Summary

This comprehensive pipeline covers:

1. **Data Exploration**: Inspecting the data, handling missing values, and exploring distributions.
2. **Feature Engineering**: Encoding categorical features, selecting important features using Chi-Square, Correlation, RFE, and LASSO.
3. **Model Training**: Using autoencoders for feature extraction, training a Random Forest Classifier, optimizing with PSO.
4. **Deep Learning**: Implementing a DNN for comparison.

Feel free to adjust the parameters and techniques according to your specific needs.