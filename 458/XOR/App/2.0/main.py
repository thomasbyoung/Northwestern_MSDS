import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import threading
import queue
import re
import os
import time
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class NeuralNetworkGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Neural Network Visualization")
        self.root.configure(bg='black')
        self.root.geometry("1800x1000")
        
        self.setup_controls()
        self.setup_layout()
        self.create_graph()
        self.create_table()
        self.create_network()

        self.queue = queue.Queue()
        self.process = None
        self.process_running = False
        self.iterations = []
        self.errors = []
        self.current_iteration = 0

    def setup_controls(self):
        self.controls_frame = tk.Frame(self.root, bg='black')
        self.controls_frame.pack(pady=10)

        self.iteration_label = tk.Label(self.controls_frame, text="Number of Iterations:", fg='white', bg='black')
        self.iteration_label.pack(side='left', padx=5)

        self.iteration_entry = tk.Entry(self.controls_frame, width=10)
        self.iteration_entry.pack(side='left', padx=5)
        self.iteration_entry.insert(0, "5000")

        self.start_button = tk.Button(self.controls_frame, text="Start", command=self.start_process, bg="white")
        self.start_button.pack(side='left', padx=10)

    def setup_layout(self):
        self.main_frame = tk.Frame(self.root, bg='black')
        self.main_frame.pack(expand=True, fill='both', padx=20, pady=20)
        
        self.network_frame = tk.Frame(self.main_frame, bg='black', width=600)
        self.network_frame.pack(side='left', expand=True, fill='both')
        
        self.graph_frame = tk.Frame(self.main_frame, bg='black', width=600)
        self.graph_frame.pack(side='left', expand=True, fill='both')
        
        self.results_frame = tk.Frame(self.main_frame, bg='black', width=600)
        self.results_frame.pack(side='right', expand=True, fill='both')

        self.canvas = tk.Canvas(self.network_frame, width=600, height=400, bg='black', highlightthickness=0)
        self.canvas.pack(expand=True, fill='both')

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

        columns = ('iteration', 'inputs', 'desired', 'actual', 'error', 'sse')
        self.table = ttk.Treeview(self.table_frame, columns=columns, show='headings')

        for col in columns:
            self.table.heading(col, text=col.capitalize())
            self.table.column(col, width=100)

        self.table.pack(expand=True, fill='both')

    def create_network(self):
        self.canvas.delete("all")
        positions = {'input': 100, 'hidden': 300, 'output': 500}
        node_radius = 20
        self.nodes = {'input': [], 'hidden': [], 'output': []}

        # Create nodes and connections
        for layer, x in positions.items():
            for i in range(2):
                y = 200 + (i - 0.5) * 80
                node = self.canvas.create_oval(
                    x-node_radius, y-node_radius,
                    x+node_radius, y+node_radius,
                    fill='white', outline='white'
                )
                self.nodes[layer].append(node)
                self.canvas.create_text(x, y, text=f"{layer[0].upper()}{i}", fill='black')

        # Create connections
        for from_layer, to_layer in [('input', 'hidden'), ('hidden', 'output')]:
            for from_node in self.nodes[from_layer]:
                coords = self.canvas.coords(from_node)
                fx = coords[0] + node_radius
                fy = (coords[1] + coords[3]) / 2
                for to_node in self.nodes[to_layer]:
                    coords = self.canvas.coords(to_node)
                    tx = coords[0]
                    ty = (coords[1] + coords[3]) / 2
                    self.canvas.create_line(fx, fy, tx, ty, fill='white')

    def update_network_weights(self, weights):
        # Update connection colors based on weights
        pass  # Implement weight visualization later

    def start_process(self):
        if self.process_running:
            return

        iterations = self.iteration_entry.get().strip()
        if not iterations.isdigit():
            messagebox.showerror("Error", "Please enter a valid number of iterations")
            return

        # Clear previous results
        for item in self.table.get_children():
            self.table.delete(item)
        
        self.iterations = []
        self.errors = []
        self.plot.clear()
        self.graph_canvas.draw()
        
        self.process_running = True
        self.start_button.config(text="Stop", command=self.stop_process)
        
        # Start neural network training in separate thread
        self.process_thread = threading.Thread(
            target=self.run_neural_network,
            args=(iterations,)
        )
        self.process_thread.daemon = True
        self.process_thread.start()

        # Start checking queue for updates
        self.root.after(100, self.check_queue)

    def stop_process(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
        self.process_running = False
        self.start_button.config(text="Start", command=self.start_process)

    def run_neural_network(self, iterations):
        try:
            startupinfo = None
            if os.name == 'nt':
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

            # Store process info
            current_data = {
                'iteration': None,
                'inputs': None,
                'desired': None,
                'actual': None,
                'error': None,
                'sse': None
            }

            self.process = subprocess.Popen(
                ['python', 'test.py', iterations],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True,
                startupinfo=startupinfo
            )

            # Read output line by line
            for line in iter(self.process.stdout.readline, ''):
                if not self.process_running:
                    break

                line = line.strip()
                
                # Skip empty lines
                if not line:
                    continue

                # Collect data
                if "Input0 =" in line:
                    input_match = re.search(r'Input0 = \s*(\d+)\s+Input1 = \s*(\d+)', line)
                    if input_match:
                        current_data['inputs'] = f"{input_match.group(1)},{input_match.group(2)}"
                
                elif "Desired Output0 =" in line:
                    desired_match = re.search(r'Desired Output0 = \s*(\d+)\s+Desired Output1 = \s*(\d+)', line)
                    if desired_match:
                        current_data['desired'] = f"{desired_match.group(1)},{desired_match.group(2)}"
                
                elif "New Output =" in line:
                    output_match = re.search(r'New Output = ([\d.]+)', line)
                    if output_match:
                        current_data['actual'] = output_match.group(1)
                
                elif "Error(0) =" in line:
                    error_match = re.search(r'Error\(0\) = ([-\d.]+),\s+Error\(1\) = ([-\d.]+)', line)
                    if error_match:
                        current_data['error'] = f"{error_match.group(1)},{error_match.group(2)}"
                
                elif "SSE Total was" in line:
                    sse_match = re.search(r'SSE Total was ([\d.]+)', line)
                    if sse_match:
                        current_data['sse'] = float(sse_match.group(1))
                        
                elif "Iteration number" in line:
                    iteration_match = re.search(r'Iteration number\s+(\d+)', line)
                    if iteration_match:
                        current_data['iteration'] = int(iteration_match.group(1))
                        
                # Send the complete data to the queue
                if current_data['iteration'] is not None and current_data['sse'] is not None:
                    self.queue.put(('update', current_data.copy()))
                    # Reset data for next iteration
                    current_data = {
                        'iteration': None,
                        'inputs': None,
                        'desired': None,
                        'actual': None,
                        'error': None,
                        'sse': None
                    }

            # Check for any errors
            stderr = self.process.stderr.read()
            if stderr:
                self.queue.put(('error', stderr))
                
        except Exception as e:
            self.queue.put(('error', str(e)))
        finally:
            self.process_running = False
            self.queue.put(('done', None))

    def check_queue(self):
        try:
            while True:
                msg_type, data = self.queue.get_nowait()
                
                if msg_type == 'update' and isinstance(data, dict):
                    iteration = data.get('iteration')
                    sse = data.get('sse')
                    
                    if iteration is not None and sse is not None:
                        self.update_graph(iteration, sse)
                        self.table.insert('', 'end', values=(
                            iteration,
                            data.get('inputs', 'N/A'),
                            data.get('desired', 'N/A'),
                            data.get('actual', 'N/A'),
                            data.get('error', 'N/A'),
                            f"{sse:.6f}"
                        ))
                        # Auto-scroll to the bottom
                        self.table.yview_moveto(1)
                        
                        # Update UI immediately
                        self.root.update_idletasks()
                
                elif msg_type == 'error':
                    messagebox.showerror("Error", f"An error occurred: {data}")
                    self.stop_process()
                
                elif msg_type == 'done':
                    self.start_button.config(text="Start", command=self.start_process)
                
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