import pandas as pd

# Path to your JSONL file
input_file = r"C:\Users\Lenovo\Desktop\My_Project\Results\results_20250918T222443Z.jsonl"
output_file = r"C:\Users\Lenovo\Desktop\My_Project\Results\results_20250918T222443Z.csv"

# Read JSONL into pandas DataFrame
df = pd.read_json(input_file, lines=True)

# Save as CSV
df.to_csv(output_file, index=False, encoding="utf-8-sig")

print(f"CSV file saved to: {output_file}")
