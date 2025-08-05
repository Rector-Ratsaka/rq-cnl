import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("mistral_rqs.csv")  

# Split into 80% train, 20% test (stratify=False as no label column is specified)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

# Save to new CSV files
train_df.to_csv("mistral_rqs_80train.csv", index=False)
test_df.to_csv("mistral_rqs_20test.csv", index=False)

print(f"Training set: {len(train_df)} RQs")
print(f"Testing set: {len(test_df)} RQs")