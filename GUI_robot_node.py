import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np
import networkx as nx
import cv2
from scipy.spatial import KDTree
import heapq
import time

class PathPlannerNode(Node):
    def __init__(self):
        super().__init__('path_planner_node')
        self.publisher_ = self.create_publisher(Float32MultiArray, 'adjusted_waypoints', 10)
        self.timer_ = self.create_timer(1.0, self.plan_and_publish)  # Plan and publish periodically

        # Path planner parameters (these could become ROS parameters later)
        self.SAFE_MARGIN = 14
        self.CURVE_RADIUS = 1
        self.THRESHOLD_ANGLE = 20
        self.map_file = 'map4.jpg'
        self.map_file_back = 'map.jpg'
        self.OFFSET_X = 19
        self.OFFSET_Y = 15
        self.SCALE_FACTOR = 0.0081

        # Data structures
        self.grafo = None
        self.free_points = None
        self.tree = None
        self.G = None
        self.image = None
        self.image_back = None
        self.safe_free_space = None

        # Planning state (for now, hardcode start and end)
        self.start_node_name = 'm'
        self.end_node_name = 'A'
        self.adjusted_waypoints = []

        self.load_map_data()

    def load_map_data(self):
        self.get_logger().info("Loading map...")
        start_time = time.time()

        # Load and process map
        self.image = cv2.imread(self.map_file, cv2.IMREAD_GRAYSCALE)
        self.image_back = cv2.imread(self.map_file_back, cv2.IMREAD_COLOR_RGB)
        _, free_space = cv2.threshold(self.image, 127, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * self.SAFE_MARGIN, 2 * self.SAFE_MARGIN))
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
            'A': (700, 148)
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
        self.get_logger().info(f"Map loaded correctly on {elapsed_time:.2f} seconds")

    def heuristica_manhattan(self, nodo_actual, nodo_objetivo):
        return 0

    def a_estrella(self, grafo, inicio, objetivo, heuristic_func=None):
        if heuristic_func is None:
            heuristic_func = self.heuristica_manhattan
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
                return self.reconstruir_camino(came_from, actual)

            for vecino, costo in grafo.obtener_vecinos(actual):
                tentative_g_score = g_score[actual] + costo

                if tentative_g_score < g_score[vecino]:
                    came_from[vecino] = actual
                    g_score[vecino] = tentative_g_score
                    f_score[vecino] = g_score[vecino] + heuristic_func(grafo.nodos[vecino], grafo.nodos[objetivo])
                    heapq.heappush(cola_prioridad, (f_score[vecino], vecino))

        return None

    def reconstruir_camino(self, came_from, actual):
        path = [actual]
        while actual in came_from:
            actual = came_from[actual]
            path.append(actual)
        return path[::-1]

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
            angle2 += 2 * np.pi

        angles = np.linspace(angle1, angle2, num_points)
        curve = np.stack([
            center[0] + curve_radius * np.sin(angles),
            center[1] + curve_radius * np.cos(angles)
        ], axis=-1)

        return curve.astype(int)

    def plan_and_publish(self):
        if self.grafo is None or self.tree is None or self.G is None:
            self.get_logger().warn("Map data not loaded yet.")
            return

        start_node = self.start_node_name
        end_node = self.end_node_name

        self.get_logger().info(f"Planning route from {start_node} to {end_node}...")
        start_time = time.time()

        # High-level planning
        camino_nodos = self.a_estrella(self.grafo, start_node, end_node)

        if not camino_nodos:
            self.get_logger().error(f"Could not find path from {start_node} to {end_node}")
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
            start_wp_node = waypoint_nodes[i]
            end_wp_node = waypoint_nodes[i + 1]
            try:
                segment = nx.shortest_path(self.G, source=start_wp_node, target=end_wp_node, weight='weight')
                full_path.extend(segment if i == 0 else segment[1:])
            except nx.NetworkXNoPath:
                self.get_logger().error(f"No path between {start_wp_node} and {end_wp_node}")
                return

        # Insert curves
        final_path = []
        for i in range(1, len(full_path) - 1):
            p_prev = np.array(full_path[i - 1])
            p_curr = np.array(full_path[i])
            p_next = np.array(full_path[i + 1])

            v1 = p_curr - p_prev
            v2 = p_next - p_curr

            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg
            if norm_v1 > 1e-6 and norm_v2 > 1e-6:
                angle = np.degrees(np.arccos(np.clip(np.dot(v1 / norm_v1, v2 / norm_v2), -1.0, 1.0)))

                if angle < (180 - self.THRESHOLD_ANGLE):
                    curve = self.insert_curve(p_prev, p_curr, p_next)
                    final_path.extend(curve.tolist())
                else:
                    final_path.append(tuple(p_curr))
            else:
                final_path.append(tuple(p_curr))

        if full_path:
            final_path = [tuple(full_path[0])] + final_path + [tuple(full_path[-1])]

            waypoints_final = [(p[1], p[0]) for p in final_path]
            reduced_waypoints = waypoints_final[::1000]
            self.adjusted_waypoints = [((wp[0] + self.OFFSET_X) * self.SCALE_FACTOR,
                                         (788 - wp[1] - self.OFFSET_Y) * self.SCALE_FACTOR)
                                        for wp in reduced_waypoints]

            msg = Float32MultiArray()
            msg.data = np.array(self.adjusted_waypoints).flatten().tolist()
            self.publisher_.publish(msg)
            self.get_logger().info(f"Published {len(self.adjusted_waypoints)} adjusted waypoints.")

            elapsed_time = time.time() - start_time
            self.get_logger().info(f"Planning completed in {elapsed_time:.2f} seconds")
            self.get_logger().info(f"Path: {' -> '.join(camino_nodos)}")
            self.get_logger().info(f"Path nodes: {len(camino_nodos)}")
            # self.get_logger().info(f"Point nodes: {len(final_path)}")
        else:
            self.get_logger().warn("No full path found.")

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

def main(args=None):
    rclpy.init(args=args)
    path_planner_node = PathPlannerNode()
    rclpy.spin(path_planner_node)
    path_planner_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()