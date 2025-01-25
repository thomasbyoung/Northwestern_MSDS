# import tkinter as tk
# from matplotlib.figure import Figure
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# class ErrorGraph:
#     def __init__(self, frame):
#         self.figure = Figure(figsize=(6, 4), facecolor='black')
#         self.plot = self.figure.add_subplot(111)
#         self.setup_plot()
        
#         self.graph_canvas = FigureCanvasTkAgg(self.figure, frame)
#         self.graph_canvas.get_tk_widget().pack(expand=True, fill='both')
        
#         self.iterations = []
#         self.errors = []
        
#     def setup_plot(self):
#         self.plot.set_facecolor('black')
#         self.plot.tick_params(colors='white')
#         self.plot.set_title('Error Over Time', color='white')
#         self.plot.set_xlabel('Iteration', color='white')
#         self.plot.set_ylabel('SSE', color='white')
#         self.plot.grid(True, color='gray', alpha=0.3)
        
#     def update(self, iteration, error):
#         self.iterations.append(iteration)
#         self.errors.append(error)
        
#         self.plot.clear()
#         self.plot.plot(self.iterations, self.errors, color='white', linewidth=1)
#         self.setup_plot()
#         self.graph_canvas.draw()