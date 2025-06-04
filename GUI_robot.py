import sys
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QGridLayout, QLabel, QPushButton, 
                            QTextEdit, QSplitter, QFrame)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
import cv2
import heapq
from scipy.spatial import KDTree
from matplotlib.figure import Figure
import time

class Nodo:
    def __init__(self, nombre, x=None, y=None):
        self.nombre = nombre
        self.x = x  
        self.y = y  
    
    def __repr__(self):
        return self.nombre

class GrafoDirigido:
    def __init__(self):
        self.nodos = {}
        self.aristas = {}
    
    def agregar_nodo(self, nodo):
        self.nodos[nodo.nombre] = nodo
        self.aristas[nodo.nombre] = {}
    
    def agregar_arista(self, origen, destino, costo=1):
        if origen in self.nodos and destino in self.nodos:
            self.aristas[origen][destino] = costo
        else:
            raise ValueError("Option does not exist.")
    
    def obtener_vecinos(self, nodo):
        return self.aristas.get(nodo, {}).items()

def heuristica_manhattan(nodo_actual, nodo_objetivo):
    return 0

def a_estrella(grafo, inicio, objetivo, heuristic_func=heuristica_manhattan):
    cola_prioridad = []
    heapq.heappush(cola_prioridad, (0, inicio))
    
    came_from = {}
    g_score = {nodo: float('inf') for nodo in grafo.nodos}
    g_score[inicio] = 0
    
    f_score = {nodo: float('inf') for nodo in grafo.nodos}
    f_score[inicio] = heuristic_func(grafo.nodos[inicio], grafo.nodos[objetivo])

    while cola_prioridad:
        _, actual = heapq.heappop(cola_prioridad)

        if actual == objetivo:
            return reconstruir_camino(came_from, actual)

        for vecino, costo in grafo.obtener_vecinos(actual):
            tentative_g_score = g_score[actual] + costo

            if tentative_g_score < g_score[vecino]:
                came_from[vecino] = actual
                g_score[vecino] = tentative_g_score
                f_score[vecino] = g_score[vecino] + heuristic_func(grafo.nodos[vecino], grafo.nodos[objetivo])
                heapq.heappush(cola_prioridad, (f_score[vecino], vecino))

    return None  

def reconstruir_camino(came_from, actual):
    path = [actual]
    while actual in came_from:
        actual = came_from[actual]
        path.append(actual)
    return path[::-1]

class MatplotlibCanvas(FigureCanvas):
    def __init__(self, parent=None, width=8, height=8, dpi=100, callback=None):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MatplotlibCanvas, self).__init__(self.fig)
        self.callback = callback
        self.mpl_connect('button_press_event', self.on_click)
        
    def on_click(self, event):
        if event.xdata is not None and event.ydata is not None and self.callback is not None:
            self.callback(event.xdata, event.ydata)

class PathPlannerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Path Planner Interface")
        self.resize(1200, 800)
        
        # Path planner parameters
        self.SAFE_MARGIN = 14
        self.CURVE_RADIUS = 1
        self.THRESHOLD_ANGLE = 20
        self.map_file = 'map4.jpg'
        self.map_file_back = 'map.jpg'
        
        # Data structures
        self.grafo = None
        self.free_points = None
        self.tree = None
        self.G = None
        self.image = None
        self.image_back = None
        self.safe_free_space = None
        
        # Selection state
        self.selection_mode = "start"  # "start" or "end"
        self.start_node = None
        self.end_node = None
        self.closest_node = None
        
        # Initialize UI
        self.initUI()
        
        # Load map data initially
        self.loadMapData()
        self.drawMap()
        
    def initUI(self):
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        # Left panel for map visualization
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_panel.setLayout(left_layout)
        
        # Create matplotlib canvas for map with click callback
        self.canvas = MatplotlibCanvas(self, width=6, height=8, callback=self.on_map_click)
        left_layout.addWidget(self.canvas)
        
        # Right panel split into top and bottom
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_panel.setLayout(right_layout)
        
        # Top right panel for controls
        control_panel = QWidget()
        control_layout = QGridLayout()
        control_panel.setLayout(control_layout)
        
        # Node selection status
        selection_frame = QFrame()
        selection_frame.setFrameShape(QFrame.StyledPanel)
        selection_layout = QGridLayout()
        selection_frame.setLayout(selection_layout)
        
        # Start node selection
        selection_layout.addWidget(QLabel("Start Node:"), 0, 0)
        self.start_node_label = QLabel("Not selected")
        self.start_node_label.setStyleSheet("font-weight: bold; color: green;")
        selection_layout.addWidget(self.start_node_label, 0, 1)
        
        # End node selection
        selection_layout.addWidget(QLabel("Final Node:"), 1, 0)
        self.end_node_label = QLabel("Not selected")
        self.end_node_label.setStyleSheet("font-weight: bold; color: red;")
        selection_layout.addWidget(self.end_node_label, 1, 1)
        
        # Selection mode buttons
        self.select_start_button = QPushButton("Select Initial")
        self.select_start_button.setCheckable(True)
        self.select_start_button.setChecked(True)
        self.select_start_button.clicked.connect(lambda: self.set_selection_mode("start"))
        selection_layout.addWidget(self.select_start_button, 2, 0)
        
        self.select_end_button = QPushButton("Select Final")
        self.select_end_button.setCheckable(True)
        self.select_end_button.clicked.connect(lambda: self.set_selection_mode("end"))
        selection_layout.addWidget(self.select_end_button, 2, 1)
        
        # Selection instruction
        selection_layout.addWidget(QLabel("Click on the map to select nodes"), 3, 0, 1, 2)
        
        # Add selection frame to control layout
        control_layout.addWidget(selection_frame, 0, 0, 1, 2)
        
        # Plan button
        self.plan_button = QPushButton("Start Planning")
        self.plan_button.clicked.connect(self.executePlanning)
        self.plan_button.setEnabled(False)  # Disabled until start and end selected
        control_layout.addWidget(self.plan_button, 2, 0)
        
        # Reset button
        self.reset_button = QPushButton("Reboot")
        self.reset_button.clicked.connect(self.resetPlanning)
        control_layout.addWidget(self.reset_button, 2, 1)
        
        # Bottom right panel for information
        info_panel = QWidget()
        info_layout = QVBoxLayout()
        info_panel.setLayout(info_layout)
        
        info_layout.addWidget(QLabel("Information:"))
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        info_layout.addWidget(self.info_text)
        
        # Add panels to right layout
        right_layout.addWidget(control_panel, 1)
        right_layout.addWidget(info_panel, 2)
        
        # Add left and right panels to main layout
        main_layout.addWidget(left_panel, 3)
        main_layout.addWidget(right_panel, 1)
        
    def set_selection_mode(self, mode):
        self.selection_mode = mode
        if mode == "start":
            self.select_start_button.setChecked(True)
            self.select_end_button.setChecked(False)
            self.log("Mode: Start node selection")
        else:
            self.select_start_button.setChecked(False)
            self.select_end_button.setChecked(True)
            self.log("Mode: End node selection")
    
    def loadMapData(self):
        self.log("Loading map...")
        start_time = time.time()
        
        # Load and process map
        self.image = cv2.imread(self.map_file, cv2.IMREAD_GRAYSCALE)
        self.image_back = cv2.imread(self.map_file_back, cv2.IMREAD_COLOR_RGB)
        _, free_space = cv2.threshold(self.image, 127, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*self.SAFE_MARGIN, 2*self.SAFE_MARGIN))
        self.safe_free_space = cv2.erode(free_space, kernel)

        self.free_points = np.column_stack(np.where(self.safe_free_space == 255))

        self.G = nx.Graph()
        for point in self.free_points:
            self.G.add_node(tuple(point))

        self.tree = KDTree(self.free_points)
        for idx, point in enumerate(self.free_points):
            distances, indices = self.tree.query(point, k=8)
            for neighbor_idx in indices[1:]:
                p1 = tuple(point)
                p2 = tuple(self.free_points[neighbor_idx])
                distance = np.linalg.norm(np.array(p1) - np.array(p2))
                self.G.add_edge(p1, p2, weight=distance)
        
        # Build high level graph
        self.grafo = GrafoDirigido()

        # Define nodes
        nodos_info = {
            'm': (39, 465), 'M': (69, 312), 'D': (150, 38), 'h': (128, 386), 'x': (110, 550),
            'H': (207, 404), 'c': (200, 454), 'C': (208, 550), 'k': (366, 38), 'K': (366, 68),
            'N': (368, 276), 'n': (368, 306), 'I': (368, 520), 'i': (368, 550), 'e': (445, 196),
            'J': (445, 368), 'f': (445, 450), 'E': (475, 196), 'j': (475, 368), 'F': (475, 450),
            'g': (575, 276), 'G': (575, 306), 'b': (575, 520), 'B': (575, 550), 'a': (712, 375), 
            'A' : (700,148)
        }

        for nombre, (x, y) in nodos_info.items():
            self.grafo.agregar_nodo(Nodo(nombre, x, y))

        # Define edges
        aristas = [
            ('a', 'B'), ('B', 'f'), ('B', 'i'), ('b', 'G'), ('C', 'x'), ('c', 'x'), ('c', 'I'),
            ('D', 'k'), ('E', 'g'), ('E', 'n'), ('E', 'j'), ('e', 'K'), ('F', 'i'), ('F', 'b'),
            ('f', 'J'), ('G', 'e'), ('G', 'j'), ('G', 'n'), ('g', 'a'), ('H', 'c'), ('h', 'N'),
            ('h', 'c'), ('I', 'f'), ('I', 'b'), ('i', 'C'), ('J', 'g'), ('J', 'e'), ('J', 'n'),
            ('j', 'F'), ('K', 'M'), ('k', 'E'), ('M', 'h'), ('m', 'D'), ('m', 'h'), ('N', 'g'),
            ('N', 'e'), ('N', 'j'), ('n', 'H'), ('x', 'm'), ('A', 'a'), ('k', 'A')
        ]

        for origen, destino in aristas:
            self.grafo.agregar_arista(origen, destino, 1)
        
        elapsed_time = time.time() - start_time
        self.log(f"Map loaded correctly on {elapsed_time:.2f} seconds")
        self.log("Click on the map to select the start node")
    
    def drawMap(self):
        # Call drawMapWithSelection which handles both normal and selection views
        self.drawMapWithSelection()
    
    def on_map_click(self, x, y):
        # Find the closest node to the click position
        closest_node = None
        min_distance = float('inf')
        
        for nombre, nodo in self.grafo.nodos.items():
            # Calculate Euclidean distance between click and node
            distance = np.sqrt((nodo.y - x)**2 + (nodo.x - y)**2)
            
            if distance < min_distance:
                min_distance = distance
                closest_node = nombre
        
        # Only select the node if it's within a reasonable distance (e.g., 30 pixels)
        if min_distance > 30:
            self.log(f"There is no node near the selected position.")
            return
        
        self.closest_node = closest_node
        
        # Update the appropriate label based on selection mode
        if self.selection_mode == "start":
            self.start_node = closest_node
            self.start_node_label.setText(closest_node)
            self.log(f"Selected start node: {closest_node}")
            
            # Automatically switch to end selection if start is selected
            if self.end_node is None:
                self.set_selection_mode("end")
        else:  # end selection
            self.end_node = closest_node
            self.end_node_label.setText(closest_node)
            self.log(f"Selected end node: {closest_node}")
        
        # Enable the planning button if both nodes are selected
        if self.start_node and self.end_node:
            self.plan_button.setEnabled(True)
        
        # Redraw the map to show selection
        self.drawMapWithSelection()
        
    def drawMapWithSelection(self):

        self.image_back = cv2.imread(self.map_file_back, cv2.IMREAD_COLOR_RGB)
        # Clear previous plot
        self.canvas.axes.clear()
        
        # Plot the map
        self.canvas.axes.imshow(self.image_back)
        
        # Plot all nodes
        for nombre, nodo in self.grafo.nodos.items():
            if nombre == self.start_node:
                # Start node in green
                self.canvas.axes.plot(nodo.y, nodo.x, 'go', markersize=10)
                self.canvas.axes.text(nodo.y+5, nodo.x+5, nombre, color='green', fontsize=12, weight='bold')
            elif nombre == self.end_node:
                # End node in red
                self.canvas.axes.plot(nodo.y, nodo.x, 'ro', markersize=10)
                self.canvas.axes.text(nodo.y+5, nodo.x+5, nombre, color='red', fontsize=12, weight='bold')
            else:
                # Other nodes in blue
                self.canvas.axes.plot(nodo.y, nodo.x, 'bo', markersize=6)
                self.canvas.axes.text(nodo.y+5, nodo.x+5, nombre, color='blue', fontsize=10)
        
        # Draw edges
        # for origen, destinos in self.grafo.aristas.items():
            # origen_nodo = self.grafo.nodos[origen]
            # for destino in destinos:
                # destino_nodo = self.grafo.nodos[destino]
                # self.canvas.axes.plot([origen_nodo.y, destino_nodo.y], 
                                     # [origen_nodo.x, destino_nodo.x], 'b-', linewidth=0.5, alpha=0.5)
        
        self.canvas.axes.set_title("Navigation Map")
        self.canvas.axes.axis('off')
        self.canvas.fig.tight_layout()
        self.canvas.draw()
    
    def executePlanning(self):
        if not self.start_node or not self.end_node:
            self.log("Error: Select start and end nodes before planning")
            return
            
        if self.start_node == self.end_node:
            self.log("Error: You must select different nodes")
            return
            
        inicio = self.start_node
        objetivo = self.end_node
        
        self.log(f"Planning route from {inicio} to {objetivo}...")
        start_time = time.time()
        
        # High-level planning
        camino_nodos = a_estrella(self.grafo, inicio, objetivo)
        
        if not camino_nodos:
            self.log(f"Error: Could not find path from {inicio} to {objetivo}")
            return
        
        # Get (x,y) of each node in path
        waypoints = [(self.grafo.nodos[n].x, self.grafo.nodos[n].y) for n in camino_nodos]
        
        # Map waypoints to nearest free space nodes
        waypoint_nodes = []
        for wp in waypoints:
            _, idx = self.tree.query(wp)
            waypoint_nodes.append(tuple(self.free_points[idx]))
        
        # Low-level path planning through waypoints
        full_path = []
        for i in range(len(waypoint_nodes) - 1):
            start_node = waypoint_nodes[i]
            end_node = waypoint_nodes[i+1]
            try:
                segment = nx.shortest_path(self.G, source=start_node, target=end_node, weight='weight')
                full_path.extend(segment if i == 0 else segment[1:])
            except nx.NetworkXNoPath:
                self.log(f"Error: No path between {start_node} and {end_node}")
                return
        
        # Insert curves
        final_path = []
        for i in range(1, len(full_path)-1):
            p_prev = np.array(full_path[i-1])
            p_curr = np.array(full_path[i])
            p_next = np.array(full_path[i+1])

            v1 = p_curr - p_prev
            v2 = p_next - p_curr

            angle = np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0)))

            if angle < (180 - self.THRESHOLD_ANGLE):
                curve = self.insert_curve(p_prev, p_curr, p_next)
                final_path.extend(curve.tolist())
            else:
                final_path.append(tuple(p_curr))

        final_path = [tuple(full_path[0])] + final_path + [tuple(full_path[-1])]
        
        waypoints_final = [(p[1], p[0]) for p in final_path] 
        reduced_waypoints = waypoints_final[::1000]
        OFFSET_X=19
        OFFSET_Y=15
        adjusted_waypoints = [((wp[0] + OFFSET_X) * 0.0081, (788-wp[1] - OFFSET_Y) * 0.0081) for wp in reduced_waypoints]

        print(adjusted_waypoints)

        # Draw path on map
        self.drawMapWithPath(final_path, waypoints, camino_nodos)
        
        elapsed_time = time.time() - start_time
        self.log(f"Planning completed in {elapsed_time:.2f} seconds")
        self.log(f"Path: {' -> '.join(camino_nodos)}")
        self.log(f"Path nodes: {len(camino_nodos)}")
        #self.log(f"Point nodes: {len(final_path)}")
    
    def insert_curve(self, p1, p2, p3, num_points=20, curve_radius=None):
        if curve_radius is None:
            curve_radius = self.CURVE_RADIUS
            
        p1 = np.array(p1)
        p2 = np.array(p2)
        p3 = np.array(p3)

        v1 = p2 - p1
        v2 = p3 - p2
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)

        bisector = v1 + v2
        bisector /= np.linalg.norm(bisector)

        center = p2 + bisector * curve_radius

        angle1 = np.arctan2(p1[0] - center[0], p1[1] - center[1])
        angle2 = np.arctan2(p3[0] - center[0], p3[1] - center[1])

        if angle2 < angle1:
            angle2 += 2*np.pi

        angles = np.linspace(angle1, angle2, num_points)
        curve = np.stack([
            center[0] + curve_radius * np.sin(angles),
            center[1] + curve_radius * np.cos(angles)
        ], axis=-1)

        return curve.astype(int)
    
    def drawMapWithPath(self, final_path, waypoints, camino_nodos):
               
        # Draw path on image
        for p1, p2 in zip(final_path[:-1], final_path[1:]):
            cv2.line(self.image_back, (p1[1], p1[0]), (p2[1], p2[0]), (0, 0, 255), 2)
        
        # Clear previous plot
        self.canvas.axes.clear()
        
        # Plot the map with path
        self.canvas.axes.imshow(self.image_back)
        
        # Plot all nodes
        for nombre, nodo in self.grafo.nodos.items():
            color = 'red' if nombre in camino_nodos else 'blue'
            size = 10 if nombre in camino_nodos else 6
            self.canvas.axes.plot(nodo.y, nodo.x, 'o', color=color, markersize=size)
            self.canvas.axes.text(nodo.y+5, nodo.x+5, nombre, color=color, fontsize=10)
        
        # Highlight waypoints
        for wp in waypoints:
            self.canvas.axes.plot(wp[1], wp[0], 'go', markersize=8)
        
        self.canvas.axes.set_title(f"Route: {' -> '.join(camino_nodos)}")
        self.canvas.axes.axis('off')
        self.canvas.fig.tight_layout()
        self.canvas.draw()
    
    def resetPlanning(self):
        self.log("Restarting planning...")
        self.drawMap()
        self.log("Planning restarted.")
    
    def log(self, message):
        self.info_text.append(message)
        # Scroll to bottom
        scrollbar = self.info_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PathPlannerApp()
    window.show()
    sys.exit(app.exec_())
