# Step 0: User Input
n = int(input("Enter number of data points: "))

x1 = list(map(float, input("Enter x1 values (space separated): ").split()))
x2 = list(map(float, input("Enter x2 values (space separated): ").split()))
y  = list(map(float, input("Enter y values  (space separated): ").split()))

# Validation
if len(x1) != n or len(x2) != n or len(y) != n:
    print("Error: Number of values must match n")
    exit()

# Step 1: Summations
Sx1 = sum(x1)
Sx2 = sum(x2)
Sy  = sum(y)

Sx1x1 = sum(i*i for i in x1)
Sx2x2 = sum(i*i for i in x2)

Sx1x2 = sum(x1[i] * x2[i] for i in range(n))
Sx1y  = sum(x1[i] * y[i]  for i in range(n))
Sx2y  = sum(x2[i] * y[i]  for i in range(n))

# Step 2: Determinants
D = (n*(Sx1x1*Sx2x2 - Sx1x2**2)
     - Sx1*(Sx1*Sx2x2 - Sx2*Sx1x2)
     + Sx2*(Sx1*Sx1x2 - Sx1x1*Sx2))

D0 = (Sy*(Sx1x1*Sx2x2 - Sx1x2**2)
      - Sx1*(Sx1y*Sx2x2 - Sx2y*Sx1x2)
      + Sx2*(Sx1y*Sx1x2 - Sx1x1*Sx2y))

D1 = (n*(Sx1y*Sx2x2 - Sx2y*Sx1x2)
      - Sy*(Sx1*Sx2x2 - Sx2*Sx1x2)
      + Sx2*(Sx1*Sx2y - Sx1y*Sx2))

D2 = (n*(Sx1x1*Sx2y - Sx1y*Sx1x2)
      - Sx1*(Sx1*Sx2y - Sx1y*Sx2)
      + Sy*(Sx1*Sx1x2 - Sx1x1*Sx2))

# Step 3: Coefficients
b0 = D0 / D
b1 = D1 / D
b2 = D2 / D

print("\nRegression Equation:")
print(f"y = {b0:.4f} + {b1:.4f}*x1 + {b2:.4f}*x2")

# Prediction
new_x1 = float(input("\nEnter x1 for prediction: "))
new_x2 = float(input("Enter x2 for prediction: "))
y_pred = b0 + b1*new_x1 + b2*new_x2

print(f"Predicted y = {y_pred:.4f}")
