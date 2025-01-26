import random
import numpy as np
import os

# Parameters
F = 3  # Number of facilities
C = 4  # Number of customers
Ds = 5  # Number of S-type drones
Dl = 5  # Number of L-type drones
grid_size = 100  # Grid size for the facility and customer locations

# Max possible values for parameters (adjust based on problem scale)
max_distance = 100
max_cost = 50
max_capacity = 20
max_payload_small = 15
max_payload_large = 30
max_range_small = 50
max_range_large = 100

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

# Generate distances between points (Euclidean distance)
def compute_distances(points):
    n = len(points)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt((points[i].x - points[j].x) ** 2 + (points[i].y - points[j].y) ** 2)
            distances[i][j] = distances[j][i] = dist
    return distances

# Generate random operational costs for drones
def generate_costs(n_facilities, n_customers):
    # Random cost between facility and customer for both types of drones
    cost_s = np.random.randint(5, 25, (n_facilities, n_customers))
    cost_l = np.random.randint(10, 35, (n_facilities, n_customers))
    return cost_s, cost_l

# Generate random facility capacities
def generate_facility_capacities(n_facilities):
    return np.random.randint(5, max_capacity, n_facilities)

# Generate random payload capacities for drones
def generate_payloads():
    return random.randint(5, max_payload_small), random.randint(10, max_payload_large)

# Generate random range values for drones
def generate_ranges():
    return random.randint(30, max_range_small), random.randint(60, max_range_large)

# Generate the problem instance
def generate_problem_instance():
    # Step 1: Generate points for facilities and customers
    facilities = generate_points(F, grid_size)
    customers = generate_points(C, grid_size)

    # Step 2: Compute distances between all facilities and customers
    all_points = facilities + customers
    distances = compute_distances(all_points)

    # Step 3: Generate random operational costs for drones
    cost_s, cost_l = generate_costs(F, C)

    # Step 4: Generate random facility capacities
    facility_capacities = generate_facility_capacities(F)

    # Step 5: Generate payload capacities for drones
    payload_small, payload_large = generate_payloads()

    # Step 6: Generate range values for drones
    range_small, range_large = generate_ranges()

    # Step 7: Format the problem instance for saving
    problem_instance = {
        'facilities': facilities,
        'customers': customers,
        'distances': distances,
        'cost_s': cost_s,
        'cost_l': cost_l,
        'facility_capacities': facility_capacities,
        'payload_small': payload_small,
        'payload_large': payload_large,
        'range_small': range_small,
        'range_large': range_large
    }

    return problem_instance

# Save problem instance to disk
def save_problem_to_disk(problem_instance, filename="problem_instance"):
    folder_name = "generated_problems"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    try:
        with open(f"{folder_name}/{filename}.txt", "w") as file:
            file.write("Facility Location Problem Instance\n")
            file.write(f"Number of facilities: {F}\n")
            file.write(f"Number of customers: {C}\n")
            file.write(f"Grid size: {grid_size}x{grid_size}\n")
            file.write("Facilities coordinates (x, y):\n")
            for facility in problem_instance['facilities']:
                file.write(f"{facility.x}, {facility.y}\n")
            file.write("Customers coordinates (x, y):\n")
            for customer in problem_instance['customers']:
                file.write(f"{customer.x}, {customer.y}\n")
            file.write("Distances between facilities and customers:\n")
            for i in range(F):
                for j in range(C):
                    file.write(f"Facility {i} to Customer {j}: {problem_instance['distances'][i][F + j]:.2f}\n")
            file.write("Drone operational costs (S-type drones):\n")
            for i in range(F):
                for j in range(C):
                    file.write(f"Facility {i} to Customer {j}: {problem_instance['cost_s'][i][j]}\n")
            file.write("Drone operational costs (L-type drones):\n")
            for i in range(F):
                for j in range(C):
                    file.write(f"Facility {i} to Customer {j}: {problem_instance['cost_l'][i][j]}\n")
            file.write("Facility capacities:\n")
            for i in range(F):
                file.write(f"Facility {i}: {problem_instance['facility_capacities'][i]}\n")
            file.write(f"Small drone payload capacity: {problem_instance['payload_small']}\n")
            file.write(f"Large drone payload capacity: {problem_instance['payload_large']}\n")
            file.write(f"Small drone max range: {problem_instance['range_small']}\n")
            file.write(f"Large drone max range: {problem_instance['range_large']}\n")
        print(f"Problem instance saved to {folder_name}/{filename}.txt")
    except Exception as e:
        print(f"Error saving problem instance: {e}")

# Generate and save multiple problem instances
def generate_and_save_multiple_problems(num_problems):
    for i in range(num_problems):
        print(f"Generating problem instance {i+1}...")
        problem_instance = generate_problem_instance()
        save_problem_to_disk(problem_instance, filename=f"problem_instance_{i+1}")

# Example: Generate and save 3 problem instances
generate_and_save_multiple_problems(10)
