import csv
import math

scores = [10, 55, 32, 70, 25, 90]

mean = sum(scores) / len(scores)
std_dev = math.sqrt(sum((x - mean) ** 2 for x in scores) / len(scores))

z_scores = [(x - mean) / std_dev for x in scores]

with open("zscore_normalized_scores.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Score", "Z_Score"])
    
    for score, z in zip(scores, z_scores):
        writer.writerow([score, z])

print("CSV file 'zscore_normalized_scores.csv' created successfully!")
