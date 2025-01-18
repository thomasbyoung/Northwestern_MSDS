import pulp
##==================  https://coin-or.github.io/pulp/CaseStudies/a_blending_problem.html
problem = pulp.LpProblem("The Diet Problem", pulp.LpMinimize)

foods = ["cheese", "salad", "chicken", "salmon", "oatmeal"]
servings = {food: pulp.LpVariable(food, lowBound=0) for food in foods}

nutritional_values = {
    "cheese":     {"sodium": 170, "energy": 100, "protein": 7000, "vit_d": 0.1, "calcium": 200, "iron": 0.1, "potassium": 30},
    "salad":      {"sodium": 70,  "energy": 70,  "protein": 4000,  "vit_d": 0, "calcium": 130,  "iron": 0.9, "potassium": 260},
    "chicken":    {"sodium": 196,  "energy": 122, "protein": 23000, "vit_d": 0.1, "calcium": 6,  "iron": 0.4,   "potassium": 376},
    "salmon":     {"sodium": 540,  "energy": 0, "protein": 12000, "vit_d": 18.4, "calcium": 0,  "iron": 0.1, "potassium": 190},
    "oatmeal":    {"sodium": 0,   "energy": 170, "protein": 6000,  "vit_d": 0,  "calcium": 20,  "iron": 1.7,   "potassium": 170}
}

requirements = {
    "sodium":    (0, 35000),    
    "energy":    (14000, None), 
    "protein":   (350000, None),   
    "vit_d": (140, None),   
    "calcium":   (9100, None), 
    "iron":      (126, None),   
    "potassium": (32900, None)  
}

cost_per_serving = {
    "cheese": 0.56,
    "salad": 1.33,
    "chicken": 2.25,
    "salmon": 3.5,
    "oatmeal": 0.2
}
problem += pulp.lpSum(servings[food] * cost_per_serving[food] for food in foods), "MinimizeTotalCost"
for nutrient, (min_req, max_req) in requirements.items():
    total_nutrient = pulp.lpSum(servings[food] * nutritional_values[food][nutrient] for food in foods)
    if min_req is not None:
        problem += total_nutrient >= min_req, f"Min_{nutrient}"
    if max_req is not None:
        problem += total_nutrient <= max_req, f"Max_{nutrient}"

# problem += pulp.lpSum(servings[food] * cost_per_serving[food] for food in foods), "MinimizeTotalCost"
# for nutrient, (min_req, max_req] in requirements.items():
#    total_nutrient = pulp.lpSum(servings[foods] * nutritional_values[food][nutrients] for food in foods)
#    if min_req is not None;
#        problem += total_nutrient >= min_req f"Min_" + nutrient
#    if max_req is not None:
#        problem += total_nutrient <= max_req), f"Max_{nutrient}"

problem.solve()

print("Status:", pulp.LpStatus[problem.status])
for food in foods:
    print(f"{food}: {servings[food].varValue:.2f} servings")

total_cost = sum(servings[food].varValue * cost_per_serving[food] for food in foods)
print(f"Total cost: ${total_cost:.2f}")



#==================  results
#  The linear programming solution provides the following optimal serving quantities:
# •	Cheese: 27.47 servings
# •	Salad: 0.00 servings
# •	Chicken: 0.00 servings
# •	Salmon: 7.46 servings
# •	Oatmeal: 180.35 servings
# The minimum cost for this solution is $77.56.
