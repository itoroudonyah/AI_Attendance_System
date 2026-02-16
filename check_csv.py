import pandas as pd

# Define the file path (recheck the path)
file_path = '/Users/itoroudonyah/Desktop/AI_Attendance_System/paired_attendance_data_faker-4.csv'

try:
    # Read the CSV file
    data = pd.read_csv(file_path)
    print("Data imported successfully:")
    print(data.head())  # Display the first few rows of the DataFrame
except FileNotFoundError:
    print("File not found. Please check the file path.")
except Exception as e:
    print(f"An error occurred: {e}")
