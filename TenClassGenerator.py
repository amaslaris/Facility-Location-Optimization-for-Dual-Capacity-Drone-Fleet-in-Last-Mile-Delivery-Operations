import random
import numpy as np
import os

# Random Seed for Reproducibility
random.seed(42)
np.random.seed(42)

# Class to represent a point (either facility or customer)
class Point:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

# Generate facilities and customers
def generate_points(n, grid_size):
    points = []
    for _ in range(n):
        x = random.randint(0, grid_size - 1)
        y = random.randint(0, grid_size - 1)
        points.append(Point(x, y))
    return points

# Generate distances between facilities and customers (Euclidean distance)
def compute_distances(facilities, customers):
    distances = np.zeros((len(facilities), len(customers)))
    for i, facility in enumerate(facilities):
        for j, customer in enumerate(customers):
            dist = np.sqrt((facility.x - customer.x) ** 2 + (facility.y - customer.y) ** 2)
            distances[i][j] = dist
    return distances

# Generate random operational costs for drones
def generate_costs(n_facilities, n_customers, cost_range_s, cost_range_l):
    cost_s = np.random.randint(*cost_range_s, (n_facilities, n_customers))
    cost_l = np.random.randint(*cost_range_l, (n_facilities, n_customers))
    return cost_s, cost_l

# Generate random facility capacities
def generate_facility_capacities(n_facilities, max_capacity):
    return np.random.randint(5, max_capacity + 1, n_facilities)

# Generate random package weights for customers
def generate_package_weights(n_customers, min_weight, max_weight):
    return np.random.randint(min_weight, max_weight + 1, n_customers)

# Generate the problem instance
def generate_problem_instance(n_facilities, n_customers, grid_size, max_capacity, cost_range_s, cost_range_l, max_payload_small, max_payload_large, max_range_small, max_range_large, package_weight_range=(1, 30)):
    facilities = generate_points(n_facilities, grid_size)
    customers = generate_points(n_customers, grid_size)
    distances = compute_distances(facilities, customers)
    cost_s, cost_l = generate_costs(n_facilities, n_customers, cost_range_s, cost_range_l)
    facility_capacities = generate_facility_capacities(n_facilities, max_capacity)
    payload_small = random.randint(5, max_payload_small)
    payload_large = random.randint(10, max_payload_large)
    range_small = random.randint(10, max_range_small)
    range_large = random.randint(20, max_range_large)
    package_weights = generate_package_weights(n_customers, *package_weight_range)

    return {
        'facilities': facilities,
        'customers': customers,
        'distances': distances,
        'cost_s': cost_s,
        'cost_l': cost_l,
        'facility_capacities': facility_capacities,
        'payload_small': payload_small,
        'payload_large': payload_large,
        'range_small': range_small,
        'range_large': range_large,
        'package_weights': package_weights,  
        'grid_size': grid_size,
        'n_facilities': n_facilities,
        'n_customers': n_customers
    }

# Save problem instance to disk (including package weights)
def save_problem_to_disk(problem_instance, class_num, problem_num):
    folder_name = f"generated_problems/class_{class_num}"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    filename = f"{folder_name}/problem_instance_{problem_num}.txt"
    with open(filename, "w") as file:
        file.write("Facility Location Problem Instance\n")
        file.write(f"Number of facilities: {problem_instance['n_facilities']}\n")
        file.write(f"Number of customers: {problem_instance['n_customers']}\n")
        file.write(f"Grid size: {problem_instance['grid_size']}x{problem_instance['grid_size']}\n")
        file.write(f"Facilities coordinates (x, y):\n")
        for facility in problem_instance['facilities']:
            file.write(f"{facility.x}, {facility.y}\n")
        file.write("Customers coordinates (x, y):\n")
        for customer in problem_instance['customers']:
            file.write(f"{customer.x}, {customer.y}\n")
        file.write("Distances between facilities and customers:\n")
        for i, facility in enumerate(problem_instance['facilities']):
            for j, customer in enumerate(problem_instance['customers']):
                file.write(f"Facility {i} to Customer {j}: {problem_instance['distances'][i][j]:.2f}\n")
        file.write("Drone operational costs (S-type drones):\n")
        for i in range(len(problem_instance['facilities'])):
            for j in range(len(problem_instance['customers'])):
                file.write(f"Facility {i} to Customer {j}: {problem_instance['cost_s'][i][j]}\n")
        file.write("Drone operational costs (L-type drones):\n")
        for i in range(len(problem_instance['facilities'])):
            for j in range(len(problem_instance['customers'])):
                file.write(f"Facility {i} to Customer {j}: {problem_instance['cost_l'][i][j]}\n")
        file.write("Facility capacities:\n")
        for i, capacity in enumerate(problem_instance['facility_capacities']):
            file.write(f"Facility {i}: {capacity}\n")
        file.write(f"Small drone payload capacity: {problem_instance['payload_small']}\n")
        file.write(f"Large drone payload capacity: {problem_instance['payload_large']}\n")
        file.write(f"Small drone max range: {problem_instance['range_small']}\n")
        file.write(f"Large drone max range: {problem_instance['range_large']}\n")
        file.write("Package weights:\n")
        for i, weight in enumerate(problem_instance['package_weights']):
            file.write(f"Customer {i}: {weight}\n")

# Generate and save multiple problem instances across classes
def generate_and_save_multiple_classes(num_problems_per_class=10):
    settings = [
    # grid_size, customers, facilities, max_cap, (cost_s), (cost_l), max_payload_s, max_payload_l, max_range_s, max_range_l
    (30, 30, 10, 50, (5, 25), (10, 35), 10, 30, 30, 60),    # Class 1 - Small instance, quick to solve (~1 second)
    (200, 12, 9, 25, (5, 30), (10, 40), 20, 35, 60, 120),    # Class 2 - Slightly larger (~5 seconds)
    (300, 15, 10, 30, (5, 35), (10, 45), 25, 40, 70, 140),   # Class 3 - Moderate size (~10 seconds)
    (400, 18, 12, 35, (5, 40), (10, 50), 30, 45, 80, 160),   # Class 4 - Larger (~20 seconds)
    (500, 20, 12, 40, (5, 45), (10, 55), 35, 50, 90, 180),   # Class 5 - Larger (~30 seconds)
    (600, 22, 14, 45, (10, 50), (15, 60), 40, 55, 100, 200),  # Class 6 - Large (~1 minute)
    (700, 25, 16, 50, (10, 55), (15, 65), 45, 60, 110, 220),  # Class 7 - Very large (~3 minutes)
    (800, 30, 18, 55, (10, 60), (15, 70), 50, 65, 120, 240),  # Class 8 - Very large (~8 minutes)
    (900, 35, 20, 60, (15, 65), (20, 75), 55, 70, 130, 260),  # Class 9 - Huge (~15 minutes)
    (1000, 400, 25, 50, (15, 70), (20, 80), 40, 80, 300, 500)  # Class 10 - Extremely large (~30 minutes)
]

    for class_num, setting in enumerate(settings, start=1):
        grid_size, n_customers, n_facilities, max_capacity, cost_range_s, cost_range_l, max_payload_s, max_payload_l, max_range_s, max_range_l = setting
        for problem_num in range(1, num_problems_per_class + 1):
            problem_instance = generate_problem_instance(
                n_facilities, n_customers, grid_size, max_capacity, cost_range_s, cost_range_l, max_payload_s, max_payload_l, max_range_s, max_range_l
            )
            save_problem_to_disk(problem_instance, class_num, problem_num)

# Generate and save problem instances for all classes
generate_and_save_multiple_classes()
