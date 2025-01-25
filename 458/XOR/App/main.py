import tkinter as tk
from tkinter import ttk
import subprocess
import threading
import queue
import re
import time
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class NeuralNetworkGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Neural Network Visualization")
        self.root.configure(bg='black')
        
        # Configure main window size
        self.root.geometry("1800x1000")

        # Add Iteration Input and Start Button
        self.controls_frame = tk.Frame(root, bg='black')
        self.controls_frame.pack(pady=10)

        self.iteration_label = tk.Label(self.controls_frame, text="Number of Iterations:", fg='white', bg='black')
        self.iteration_label.pack(side='left', padx=5)

        self.iteration_entry = tk.Entry(self.controls_frame, width=10)
        self.iteration_entry.pack(side='left', padx=5)
        self.iteration_entry.insert(0, "5000")  # Default value

        self.start_button = tk.Button(self.controls_frame, text="Start", command=self.start_process, bg="white")
        self.start_button.pack(side='left', padx=10)

        # Frames for different sections
        self.main_frame = tk.Frame(root, bg='black')
        self.main_frame.pack(expand=True, fill='both', padx=20, pady=20)
        
        # Left frame for network visualization
        self.network_frame = tk.Frame(self.main_frame, bg='black', width=600)
        self.network_frame.pack(side='left', expand=True, fill='both')
        
        # Middle frame for graph
        self.graph_frame = tk.Frame(self.main_frame, bg='black', width=600)
        self.graph_frame.pack(side='left', expand=True, fill='both')
        
        # Right frame for results
        self.results_frame = tk.Frame(self.main_frame, bg='black', width=600)
        self.results_frame.pack(side='right', expand=True, fill='both')

        # Canvas for neural network visualization
        self.canvas = tk.Canvas(self.network_frame, width=600, height=400, bg='black', highlightthickness=0)
        self.canvas.pack(expand=True, fill='both')

        # Create graph, table, and network
        self.create_graph()
        self.create_table()
        self.create_network()

        # Queue for communication between threads
        self.queue = queue.Queue()
        self.process_running = False
        self.current_iteration = {}
        self.iterations = []
        self.errors = []

    def create_graph(self):
        self.figure = Figure(figsize=(6, 4), facecolor='black')
        self.plot = self.figure.add_subplot(111)
        self.plot.set_facecolor('black')
        self.plot.tick_params(colors='white')
        self.plot.set_title('Error Over Time', color='white')
        self.plot.set_xlabel('Iteration', color='white')
        self.plot.set_ylabel('SSE', color='white')
        self.plot.grid(True, color='gray', alpha=0.3)

        self.graph_canvas = FigureCanvasTkAgg(self.figure, self.graph_frame)
        self.graph_canvas.get_tk_widget().pack(expand=True, fill='both')

    def update_graph(self, iteration, error):
        self.iterations.append(iteration)
        self.errors.append(error)

        self.plot.clear()
        self.plot.plot(self.iterations, self.errors, color='white', linewidth=1)
        self.plot.set_facecolor('black')
        self.plot.tick_params(colors='white')
        self.plot.set_title('Error Over Time', color='white')
        self.plot.set_xlabel('Iteration', color='white')
        self.plot.set_ylabel('SSE', color='white')
        self.plot.grid(True, color='gray', alpha=0.3)
        self.graph_canvas.draw()

    def create_table(self):
        style = ttk.Style()
        style.theme_use('default')
        style.configure("Treeview", background="black", foreground="white", fieldbackground="black")
        style.configure("Treeview.Heading", background="black", foreground="white")

        self.table_frame = tk.Frame(self.results_frame, bg='black')
        self.table_frame.pack(expand=True, fill='both')

        self.table_scroll = ttk.Scrollbar(self.table_frame)
        self.table_scroll.pack(side='right', fill='y')

        columns = ('iteration', 'inputs', 'desired', 'actual', 'error', 'sse')
        self.table = ttk.Treeview(self.table_frame, columns=columns, show='headings',
                                  yscrollcommand=self.table_scroll.set, style="Treeview")

        for col in columns:
            self.table.heading(col, text=col.capitalize())
            self.table.column(col, width=120)

        self.table.pack(expand=True, fill='both')
        self.table_scroll.config(command=self.table.yview)

    def create_network(self):
        self.canvas.delete("all")
        positions = {'input': 100, 'hidden': 300, 'output': 500}
        node_radius = 20
        self.nodes = {'input': [], 'hidden': [], 'output': []}

        for layer, x in positions.items():
            for i in range(2):
                y = 200 + (i - 0.5) * 80
                node = self.canvas.create_oval(x-node_radius, y-node_radius, x+node_radius, y+node_radius, fill='white', outline='white')
                self.nodes[layer].append(node)
                self.canvas.create_text(x, y, text=f"{layer[0].upper()}{i}", fill='black')

        for from_layer, to_layer in [('input', 'hidden'), ('hidden', 'output')]:
            for from_node in self.nodes[from_layer]:
                fx, fy, _, _ = self.canvas.coords(from_node)
                fx, fy = fx + node_radius, (fy + fy + node_radius) / 2
                for to_node in self.nodes[to_layer]:
                    tx, ty, _, _ = self.canvas.coords(to_node)
                    tx, ty = tx - node_radius, (ty + ty + node_radius) / 2
                    self.canvas.create_line(fx, fy, tx, ty, fill='white')

    def start_process(self):
        if not self.process_running:
            iterations = self.iteration_entry.get().strip()
            if not iterations.isdigit():
                print("Invalid iteration count. Using default: 5000")
                iterations = "5000"

            self.process_running = True
            self.process_thread = threading.Thread(target=self.run_neural_network, args=(iterations,))
            self.process_thread.daemon = True
            self.process_thread.start()
            self.root.after(100, self.check_queue)

    def run_neural_network(self, iterations):
        try:
            process = subprocess.Popen(['python', 'test.py', iterations], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       universal_newlines=True, bufsize=1)

            for line in process.stdout:
                print(line.strip())  # Debugging output

            process.wait()
        except Exception as e:
            print(f"Error running test.py: {e}")
        finally:
            self.process_running = False

    def check_queue(self):
        try:
            while not self.queue.empty():
                item = self.queue.get_nowait()
                self.queue.task_done()
        except queue.Empty:
            pass
        finally:
            if self.process_running:
                self.root.after(100, self.check_queue)

if __name__ == "__main__":
    root = tk.Tk()
    app = NeuralNetworkGUI(root)
    root.mainloop()
