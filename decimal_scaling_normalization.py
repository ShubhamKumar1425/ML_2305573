import csv

values = [120, 45, 300, 12, 8, 950]

max_value = max(abs(x) for x in values)
j = len(str(max_value))

decimal_scaled = [x / (10 ** j) for x in values]

with open("decimal_scaled_output.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Value", "Decimal_Scaled_Value"])
    
    for v, ds in zip(values, decimal_scaled):
        writer.writerow([v, ds])

print("CSV file 'decimal_scaled_output.csv' created successfully!")
