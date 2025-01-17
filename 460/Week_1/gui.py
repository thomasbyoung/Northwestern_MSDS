import json
import tkinter as tk
from tkinter import ttk, messagebox
import pulp
import os

class NutritionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Nutrition Optimization")

        self.food_items = {}
        self.load_foods()

        self.create_widgets()

    def create_widgets(self):
        frame = tk.Frame(self.root)
        frame.pack(pady=10)
        tk.Label(frame, text="Food Name:").grid(row=0, column=0, padx=5, pady=5)
        self.food_name_entry = tk.Entry(frame)
        self.food_name_entry.grid(row=0, column=1, padx=5, pady=5)

        nutrients = ["sodium", "energy", "protein", "vitamin_d", "calcium", "iron", "potassium", "cost"]
        self.entries = {}
        for i, nutrient in enumerate(nutrients):
            tk.Label(frame, text=f"{nutrient.capitalize()}:").grid(row=i+1, column=0, padx=5, pady=5)
            self.entries[nutrient] = tk.Entry(frame)
            self.entries[nutrient].grid(row=i+1, column=1, padx=5, pady=5)

        tk.Button(frame, text="Add Food", command=self.add_food).grid(row=len(nutrients)+1, column=0, padx=5, pady=10)
        tk.Button(frame, text="Save Foods", command=self.save_foods).grid(row=len(nutrients)+1, column=1, padx=5, pady=10)
        tk.Button(frame, text="Optimize Nutrition", command=self.optimize_nutrition).grid(row=len(nutrients)+2, column=0, columnspan=2, pady=10)
        tk.Button(frame, text="Refresh", command=self.refresh_data).grid(row=len(nutrients)+3, column=0, columnspan=2, pady=10)

        self.table = ttk.Treeview(self.root, columns=["food_name"] + nutrients, show="headings", height=10)
        self.table.heading("food_name", text="Food Name")
        self.table.column("food_name", width=100, anchor="center")
        for nutrient in nutrients:
            self.table.heading(nutrient, text=nutrient.capitalize())
            self.table.column(nutrient, width=100, anchor="center")

        self.table.bind("<Double-1>", self.edit_selected)
        self.table.pack(pady=10)
        tk.Button(self.root, text="Delete Selected", command=self.delete_selected).pack(pady=5)
        constraints_frame = tk.LabelFrame(self.root, text="Nutritional Constraints", padx=10, pady=10)
        constraints_frame.pack(pady=10, fill="both", expand="yes")
        self.constraints_text = tk.Text(constraints_frame, height=10, width=60, state="disabled")
        self.constraints_text.pack()
        self.display_constraints()

        self.update_table()

    def add_food(self):
        food_name = self.food_name_entry.get().strip()
        if not food_name:
            messagebox.showerror("Error", "Food name cannot be empty!")
            return

        food_data = {}
        try:
            for nutrient, entry in self.entries.items():
                food_data[nutrient] = float(entry.get())
        except ValueError:
            messagebox.showerror("Error", "All nutrient values must be numbers!")
            return

        self.food_items[food_name] = food_data
        self.update_table()
        self.clear_entries()

    def update_table(self):
        for row in self.table.get_children():
            self.table.delete(row)
        for food, data in self.food_items.items():
            self.table.insert("", "end", values=(food, *data.values()))

    def clear_entries(self):
        self.food_name_entry.delete(0, tk.END)
        for entry in self.entries.values():
            entry.delete(0, tk.END)

    def save_foods(self):
        with open("foods.json", "w") as f:
            json.dump(self.food_items, f, indent=4)
        messagebox.showinfo("Saved", "Foods saved to foods.json!")

    def load_foods(self):
        if os.path.exists("foods.json"):
            with open("foods.json", "r") as f:
                self.food_items = json.load(f)

    def refresh_data(self):
        self.load_foods()
        self.update_table()

    def delete_selected(self):
        selected_item = self.table.selection()
        if selected_item:
            item_values = self.table.item(selected_item[0], "values")
            food_name = item_values[0]
            del self.food_items[food_name]
            self.update_table()

    def edit_selected(self, event):
        selected_item = self.table.selection()
        if selected_item:
            item_values = self.table.item(selected_item[0], "values")
            food_name = item_values[0]
            self.food_name_entry.delete(0, tk.END)
            self.food_name_entry.insert(0, food_name)

            for i, nutrient in enumerate(["sodium", "energy", "protein", "vitamin_d", "calcium", "iron", "potassium", "cost"]):
                self.entries[nutrient].delete(0, tk.END)
                self.entries[nutrient].insert(0, item_values[i+1])
            del self.food_items[food_name]

    def optimize_nutrition(self):
        if not self.food_items:
            messagebox.showerror("Error", "No food data to optimize!")
            return
        problem = pulp.LpProblem("NutritionOptimization", pulp.LpMinimize)
        servings = {food: pulp.LpVariable(food, lowBound=0) for food in self.food_items.keys()}
        problem += pulp.lpSum(servings[food] * self.food_items[food]["cost"] for food in self.food_items.keys())

        requirements = {
            "sodium":    (0, 5000),
            "energy":    (2000, None),
            "protein":   (50, None),
            "vitamin_d": (20, None),
            "calcium":   (1300, None),
            "iron":      (18, None),
            "potassium": (4700, None)
        }

        for nutrient, (min_req, max_req) in requirements.items():
            total_nutrient = pulp.lpSum(servings[food] * self.food_items[food][nutrient] for food in self.food_items.keys())
            if min_req is not None:
                problem += total_nutrient >= min_req
            if max_req is not None:
                problem += total_nutrient <= max_req

        problem.solve()

        if problem.status == 1:
            result = "\nOptimal solution:\n"
            total_cost = 0
            for food in self.food_items.keys():
                servings_val = servings[food].varValue or 0
                result += f"{food}: {servings_val:.2f} servings\n"
                total_cost += servings_val * self.food_items[food]["cost"]

            result += f"Total cost: ${total_cost:.2f}"
        else:
            result = "No optimal solution found."

        messagebox.showinfo("Optimization Result", result)

    def display_constraints(self):
        requirements = {
            "sodium":    (0, 5000),
            "energy":    (2000, None),
            "protein":   (50, None),
            "vitamin_d": (20, None),
            "calcium":   (1300, None),
            "iron":      (18, None),
            "potassium": (4700, None)
        }

        constraints_text = "Nutritional Constraints:\n"
        for nutrient, (min_req, max_req) in requirements.items():
            constraints_text += f"{nutrient.capitalize()}: Min {min_req}"
            if max_req is not None:
                constraints_text += f", Max {max_req}"
            constraints_text += "\n"

        self.constraints_text.configure(state="normal")
        self.constraints_text.delete(1.0, tk.END)
        self.constraints_text.insert(tk.END, constraints_text)
        self.constraints_text.configure(state="disabled")

if __name__ == "__main__":
    root = tk.Tk()
    app = NutritionApp(root)
    root.mainloop()
