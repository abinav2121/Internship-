import pandas as pd
import matplotlib.pyplot as plt

# Step 4: Read the CSV file into a pandas DataFrame
file_path = r'C:\Users\Hp\Documents\data.csv'  
df = pd.read_csv(file_path)

# Step 5: Understand how the code works
# Let's print the first few rows to understand the structure of your data
print(df.head())

# Step 6: Plot the graph
# Assuming you want to plot a simple line graph
plt.plot(df['X_column'], df['Y_column'])  # Replace 'X_column' and 'Y_column' with your actual column names
plt.xlabel('X Axis Label')
plt.ylabel('Y Axis Label')
plt.title('Linear Regression')
plt.grid(True)
plt.show()

# Step 7: You have completed it successfully!
