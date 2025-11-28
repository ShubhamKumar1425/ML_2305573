import csv
import random

headers = ["ID", "Age", "Salary", "Score", "Rating"]

with open("sample_data.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(headers) 

    for _ in range(10):
        row = [
            random.randint(1, 100),
            random.randint(18, 60),
            random.randint(20000, 80000),
            random.randint(1, 100),
            round(random.random(), 2)
        ]
        writer.writerow(row)

print("CSV file created successfully!")
print("Headers:", headers)