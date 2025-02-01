#  https://sgtautotransport.com/autoblog/useful-information/what-is-the-average-car-shipping-cost-per-mile-in-2025#:~:text=Average%20Cost%20to%20Ship%20a%20Car%20per%20Mile,-When%20considering%20the&text=The%20average%20cost%20per%20mile,suits%20your%20needs%20and%20budget.
# assuming $1 per mile
# Newark to Boston = 226 miles
# Newark to Columbus = 523 miles
# Newark to Atlanta = 863 miles
# Newark to Richmond = 330 miles
# Newark to Mobile = 1203 miles

# Jacksonville to Boston = 1149 miles
# Jacksonville to Columbus = 807 miles
# Jacksonville to Atlanta = 345 miles
# Jacksonville to Richmond = 598 miles
# Jacksonville to Mobile = 404 miles

import pulp

ports = ["Newark", "Jacksonville"]
distributors = ["Boston", "Columbus", "Atlanta", "Richmond", "Mobile"]
supply = {"Newark": 200, "Jacksonville": 300}
demand = {"Boston": 100, "Columbus": 60, "Atlanta": 170, "Richmond": 80, "Mobile": 70}

costs = {
    ("Newark", "Boston"): 226,
    ("Newark", "Columbus"): 523,
    ("Newark", "Atlanta"): 863,
    ("Newark", "Richmond"): 330,
    ("Newark", "Mobile"): 1203,
    ("Jacksonville", "Boston"): 1149,
    ("Jacksonville", "Columbus"): 807,
    ("Jacksonville", "Atlanta"): 345,
    ("Jacksonville", "Richmond"): 598,
    ("Jacksonville", "Mobile"): 404
}

model = pulp.LpProblem("Car_Transport", pulp.LpMinimize)
x = pulp.LpVariable.dicts("Ship", (ports, distributors), lowBound=0)
model += pulp.lpSum(costs[(i, j)] * x[i][j] for i in ports for j in distributors)
for i in ports:
    model += pulp.lpSum(x[i][j] for j in distributors) <= supply[i], f"Supply_{i}"
for j in distributors:
    model += pulp.lpSum(x[i][j] for i in ports) == demand[j], f"Demand_{j}"
model.solve(pulp.PULP_CBC_CMD(msg=False))
print(f"Status: {pulp.LpStatus[model.status]}")
print(f"Optimal Cost: {pulp.value(model.objective):,.2f}\n")

for i in ports:
    for j in distributors:
        qty = x[i][j].varValue
        if qty > 0:
            print(f"Ship {qty:.0f} cars from {i} to {j}")
