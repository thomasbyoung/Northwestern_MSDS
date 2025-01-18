import pulp
from pulp import LpVariable, LpProblem, LpMaximize, LpStatus, value, LpMinimize

print("First Linear Programming Problem (can have 0 values for certain items):")


cheese = LpVariable("cheese", 0, None)
salad = LpVariable("salad", 0, None)
chicken = LpVariable("chicken", 0, None)
salmon = LpVariable("salmon", 0, None)
oatmeal = LpVariable("oatmeal", 0, None)

prob = LpProblem("problem", LpMinimize)

prob += 170*cheese + 70*salad + 196*chicken + 540*salmon + 0*oatmeal <= 35000 
prob += 100*cheese + 70*salad + 122*chicken + 0*salmon + 170*oatmeal >= 14000 
prob += 7000*cheese + 4000*salad + 23000*chicken + 12000*salmon + 6000*oatmeal >= 350000 
prob += 0.1*cheese + 0*salad + 0.1*chicken + 18.4*salmon + 0*oatmeal >= 140 
prob += 200*cheese + 130*salad + 6*chicken + 0*salmon + 20*oatmeal >= 9100 
prob += 0.1*cheese + 0.9*salad + 0.4*chicken + 0.1*salmon + 1.7*oatmeal >= 126  
prob += 30*cheese + 260*salad + 376*chicken + 190*salmon + 170*oatmeal >= 32900  

prob += 0.56*cheese + 1.33*salad + 2.25*chicken + 3.5*salmon + 0.2*oatmeal

status = prob.solve()
print(f"Problem")

for variable in prob.variables():
    print(f"{variable.name} = {variable.varValue}")
    
print(f"Objective = {value(prob.objective)}")

print("\n\n ----------------------------------- \n\n")
print("Second Linear Programming Problem (minimum 1 serving of each item):")

cheese = LpVariable("cheese", 1, None)
salad = LpVariable("salad", 1, None)
chicken = LpVariable("chicken", 1, None)
salmon = LpVariable("salmon", 1, None)
oatmeal = LpVariable("oatmeal", 1, None)

prob = LpProblem("problem", LpMinimize)

prob += 170*cheese + 70*salad + 196*chicken + 540*salmon + 0*oatmeal <= 35000  
prob += 100*cheese + 70*salad + 122*chicken + 0*salmon + 170*oatmeal >= 14000 
prob += 7000*cheese + 4000*salad + 23000*chicken + 12000*salmon + 6000*oatmeal >= 350000 
prob += 0.1*cheese + 0*salad + 0.1*chicken + 18.4*salmon + 0*oatmeal >= 140  
prob += 200*cheese + 130*salad + 6*chicken + 0*salmon + 20*oatmeal >= 9100  
prob += 0.1*cheese + 0.9*salad + 0.4*chicken + 0.1*salmon + 1.7*oatmeal >= 126  
prob += 30*cheese + 260*salad + 376*chicken + 190*salmon + 170*oatmeal >= 32900  

prob += 0.56*cheese + 1.33*salad + 2.25*chicken + 3.5*salmon + 0.2*oatmeal

status = prob.solve()
print(f"Problem with minimum 1 serving")

for variable in prob.variables():
    print(f"{variable.name} = {variable.varValue}")
    
print(f"Objective = {value(prob.objective)}")