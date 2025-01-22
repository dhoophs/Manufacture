import pandas as pd
import numpy as np

# Parameters for data generation
num_records = 10000  # Number of rows in the dataset
machine_ids = ['M1', 'M2', 'M3', 'M4', 'M5']  # List of machine IDs

# Generate synthetic data
np.random.seed(42)  # For reproducibility
data = {
    "Machine_ID": np.random.choice(machine_ids, size=num_records),
    "Temperature": np.random.uniform(50, 120, size=num_records).round(2),  # Random temperatures
    "Run_Time": np.random.uniform(100, 500, size=num_records).round(2),    # Random run times
    "Downtime_Flag": np.random.choice([0, 1], size=num_records, p=[0.8, 0.2])  # 20% downtime
}

# Create a DataFrame
df = pd.DataFrame(data)

# Save to a CSV file
df.to_csv("synthetic_manufacturing_data.csv", index=False)

print("Synthetic dataset generated and saved as 'synthetic_manufacturing_data.csv'")
