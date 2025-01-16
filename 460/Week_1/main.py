import pulp

nutritional_values = {
    "cheese":     {"sodium": 170, "energy": 100, "protein": 7000, "vitamin_d": 0.1, "calcium": 200, "iron": 0.1, "potassium": 30},
    "salad":      {"sodium": 70,  "energy": 70,  "protein": 4000,  "vitamin_d": 0, "calcium": 130,  "iron": 0.9, "potassium": 260},
    "chicken":    {"sodium": 196,  "energy": 122, "protein": 23000, "vitamin_d": 0.1, "calcium": 6,  "iron": 0.4,   "potassium": 376},
    "salmon":     {"sodium": 540,  "energy": 0, "protein": 12000, "vitamin_d": 18.4, "calcium": 0,  "iron": 0.1, "potassium": 190},
    "oatmeal":    {"sodium": 0,   "energy": 170, "protein": 6000,  "vitamin_d": 0,  "calcium": 20,  "iron": 1.7,   "potassium": 170}
}

requirements = {
    "sodium":    (0, 5000),    
    "energy":    (2000, None), 
    "protein":   (50000, None),   
    "vitamin_d": (20, None),   
    "calcium":   (1300, None), 
    "iron":      (18, None),   
    "potassium": (4700, None)  
}