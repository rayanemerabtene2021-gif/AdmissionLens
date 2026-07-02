import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# 1. Generate synthetic dataset for demonstration purposes
# In a real scenario, this would be loaded from an external source
data = {'Months': [1, 2, 3, 4, 5, 6], 'Applications': [100, 150, 200, 250, 300, 350]}
df = pd.DataFrame(data)

# Save the dataset as a CSV file for auditing and transparency
df.to_csv('admissions_data.csv', index=False)

# 2. Train a simple Linear Regression model to forecast future trends
X = df[['Months']]
y = df['Applications']
model = LinearRegression().fit(X, y)

# Predict the number of applications for the next month (Month 7)
prediction = model.predict([[7]])
print(f"Predicted applications for month 7: {prediction[0]}")

# 3. Visualize the findings and save the plot as an image
plt.scatter(df['Months'], df['Applications'], color='blue', label='Actual Data')
plt.plot(df['Months'], model.predict(X), color='red', label='Prediction Trend')
plt.title('University Admission Trends Analysis')
plt.xlabel('Month')
plt.ylabel('Applications Volume')
plt.legend()

# Export the visualization to PNG format for report presentation
plt.savefig('visualization.png')
print("Pipeline executed successfully. Outputs: admissions_data.csv and visualization.png created.")
