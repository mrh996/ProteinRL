import csv

# Define file paths
input_txt_file = '/LOCAL2/mur/MRH/protein_RL/prediction_model/common/code/baye_random_large'
output_csv_file = "/LOCAL2/mur/MRH/protein_RL/prediction_model/common/code/baye_random_large.csv"

# Open the .txt file and write to .csv file
with open(input_txt_file, 'r') as txt_file, open(output_csv_file, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)

    for line in txt_file:
        # Split the line into a list based on the delimiter (e.g., space, comma, or tab)
        row = line.strip().split(',')  # Use `.split(',')` for comma-separated, or `.split('\t')` for tab-separated
        csv_writer.writerow(row)

print(f"Data has been converted to {output_csv_file}")