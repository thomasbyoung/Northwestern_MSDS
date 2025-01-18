from pulp import *

# Define the problem
prob = LpProblem("Diet Problem", LpMinimize)

# Define the decision variables (servings of each food item)
cheese = LpVariable("Cheese", 0)
salad = LpVariable("Salad", 0)
chicken = LpVariable("Chicken", 0)
salmon = LpVariable("Salmon", 0)
oatmeal = LpVariable("Oatmeal", 0)

# Define the objective function (minimize total cost)
prob += 0.56 * cheese + 1.33 * salad + 2.25 * chicken + 3.5 * salmon + 0.2 * oatmeal

# Define the constraints
# Sodium constraint (max 5000 mg)
prob += 170 * cheese + 70 * salad + 196 * chicken + 540 * salmon + 0 * oatmeal <= 5000

# Energy constraint (min 2000 kcal)
prob += 100 * cheese + 70 * salad + 122 * chicken + 0 * salmon + 170 * oatmeal >= 2000

# Protein constraint (min 50 g)
prob += 7000 * cheese + 4000 * salad + 23000 * chicken + 12000 * salmon + 6000 * oatmeal >= 50

# Vitamin D constraint (min 20 mcg)
prob += 0.1 * cheese + 0 * salad + 0.1 * chicken + 18.4 * salmon + 0 * oatmeal >= 20

# Calcium constraint (min 1300 mg)
prob += 200 * cheese + 130 * salad + 6 * chicken + 0 * salmon + 20 * oatmeal >= 1300

# Iron constraint (min 18 mg)
prob += 0.1 * cheese + 0.9 * salad + 0.4 * chicken + 0.1 * salmon + 1.7 * oatmeal >= 18

# Potassium constraint (min 4700 mg)
prob += 30 * cheese + 260 * salad + 376 * chicken + 190 * salmon + 170 * oatmeal >= 4700

# Solve the problem
prob.solve()

# Print the status of the solution
print("Status:", LpStatus[prob.status])

# Print the optimal solution
for v in prob.variables():
    if v.varValue > 0:
        print(v.name, "=", v.varValue)

# Print the minimum cost
print("Total Cost = ", value(prob.objective))