import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import sys

# This script expects a CSV file as an input argument
# Usage: python model.py your_data.csv

def analyze_university_data(file_path):
    try:
        # Load the data provided by the university
        df = pd.read_csv(file_path)
        
        # Ensure the CSV has columns: 'Months' and 'Applications'
        X = df[['Months']]
        y = df['Applications']
        
        # Train the model on the university's actual data
        model = LinearRegression().fit(X, y)
        
        # Visualize the university's data and the trend line
        plt.scatter(df['Months'], df['Applications'], color='blue', label='Actual Data')
        plt.plot(df['Months'], model.predict(X), color='red', label='Trend Line')
        plt.title('Admission Trends Analysis')
        plt.xlabel('Month')
        plt.ylabel('Applications')
        plt.legend()
        plt.savefig('university_analysis.png')
        print("Analysis completed successfully. 'university_analysis.png' generated.")
        
    except Exception as e:
        print(f"Error processing file: {e}")

if __name__ == "__main__":
    # If a file is provided, use it. Otherwise, notify the user.
    if len(sys.argv) > 1:
        analyze_university_data(sys.argv[1])
    else:
        print("Please provide a CSV file. Usage: python model.py data.csv")
