import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns


np.random.seed(42)
n_samples = 10000


distances = np.random.uniform(0.5, 5.0, n_samples) # 0.5km to 5km radius
traffic = np.random.choice(['Low', 'Moderate', 'High'], n_samples, p=[0.5, 0.3, 0.2])
weather = np.random.choice(['Clear', 'Rain', 'Fog'], n_samples, p=[0.75, 0.15, 0.10])
prep_time = np.random.uniform(1.0, 8.0, n_samples) # Time taken to pack the order
partner_rating = np.random.uniform(3.5, 5.0, n_samples)


base_risk = (distances * 0.08) + (prep_time * 0.03)


traffic_penalty = np.where(traffic == 'High', 0.35, np.where(traffic == 'Moderate', 0.15, 0))
weather_penalty = np.where(weather == 'Rain', 0.40, np.where(weather == 'Fog', 0.25, 0))
rating_penalty = (5.0 - partner_rating) * 0.05 # Lower rated partners slightly slower

total_risk_score = base_risk + traffic_penalty + weather_penalty + rating_penalty


is_late = np.where(total_risk_score + np.random.normal(0, 0.1, n_samples) > 0.85, 1, 0)

# Create DataFrame
df = pd.DataFrame({
    'Distance_km': distances,
    'Traffic_Level': traffic,
    'Weather': weather,
    'Prep_Time_min': prep_time,
    'Partner_Rating': partner_rating,
    'Is_Late': is_late
})

print("Dataset Preview:")
print(df.head(), "\n")
print(f"Total Late Orders in Dataset: {df['Is_Late'].sum()} out of {n_samples}\n")


le_traffic = LabelEncoder()
le_weather = LabelEncoder()

df['Traffic_Level_Encoded'] = le_traffic.fit_transform(df['Traffic_Level'])
df['Weather_Encoded'] = le_weather.fit_transform(df['Weather'])


X = df[['Distance_km', 'Traffic_Level_Encoded', 'Weather_Encoded', 'Prep_Time_min', 'Partner_Rating']]
y = df['Is_Late']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1] # Probability of being late

print("--- Model Evaluation Metrics ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['On Time', 'Late']))


feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\n--- Feature Importance ---")
print(feature_importances)


plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances, palette='viridis', hue='Feature', legend=False)
plt.title('What Drives Late Deliveries at Blinkit?')
plt.xlabel('Impact on Prediction (0 to 1)')
plt.ylabel('Delivery Factors')
plt.tight_layout()
plt.show()
