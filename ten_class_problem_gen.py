import random
import numpy as np
import os

# Parameters
num_classes = 10  # Number of difficulty classes
problems_per_class = 10  # Number of problems per class
base_F = 3  # Base number of facilities
base_C = 4  # Base number of customers
base_Ds = 5  # Base number of S-type drones
base_Dl = 5  # Base number of L-type drones
base_grid_size = 100  # Base grid size for facility and customer locations

# Max possible values for parameters (adjust based on problem scale)
max_distance = 100
max_cost = 50
max_capacity = 20
max_payload_small = 15
max_payload_large = 30
max_range_small = 50
max_range_large = 100

# Adjusted grid sizes for difficulty classes
grid_sizes = {
    1: 30, 2: 40, 3: 45, 4: 50, 5: 60,
    6: 65, 7: 75, 8: 80, 9: 90, 10: 100
}

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
    cost_s = np.random.randint(5, 25, (n_facilities, n_customers))
    cost_l = np.random.randint(10, 35, (n_facilities, n_customers))
    return cost_s, cost_l

# Generate random facility capacities
def generate_facility_capacities(n_facilities, max_capacity):
    return np.random.randint(5, max_capacity, n_facilities)

# Generate random payload capacities for drones
def generate_payloads(max_payload_small, max_payload_large):
    return random.randint(5, max_payload_small), random.randint(10, max_payload_large)

# Generate random range values for drones
def generate_ranges(max_range_small, max_range_large):
    return random.randint(30, max_range_small), random.randint(60, max_range_large)

# Generate the problem instance
def generate_problem_instance(F, C, Ds, Dl, grid_size, max_capacity, max_payload_small, max_payload_large, max_range_small, max_range_large):
    facilities = generate_points(F, grid_size)
    customers = generate_points(C, grid_size)
    all_points = facilities + customers
    distances = compute_distances(all_points)
    cost_s, cost_l = generate_costs(F, C)
    facility_capacities = generate_facility_capacities(F, max_capacity)
    payload_small, payload_large = generate_payloads(max_payload_small, max_payload_large)
    range_small, range_large = generate_ranges(max_range_small, max_range_large)

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
            file.write(f"Number of facilities: {len(problem_instance['facilities'])}\n")
            file.write(f"Number of customers: {len(problem_instance['customers'])}\n")
            file.write(f"Grid size: {base_grid_size}x{base_grid_size}\n")
            file.write("Facilities coordinates (x, y):\n")
            for facility in problem_instance['facilities']:
                file.write(f"{facility.x}, {facility.y}\n")
            file.write("Customers coordinates (x, y):\n")
            for customer in problem_instance['customers']:
                file.write(f"{customer.x}, {customer.y}\n")
            file.write("Distances between facilities and customers:\n")
            for i in range(len(problem_instance['facilities'])):
                for j in range(len(problem_instance['customers'])):
                    dist_index = len(problem_instance['facilities']) + j
                    if dist_index < len(problem_instance['distances']):
                        file.write(f"Facility {i} to Customer {j}: {problem_instance['distances'][i][dist_index]:.2f}\n")
            file.write("Drone operational costs (S-type drones):\n")
            for i in range(len(problem_instance['facilities'])):
                for j in range(len(problem_instance['customers'])):
                    file.write(f"Facility {i} to Customer {j}: {problem_instance['cost_s'][i][j]}\n")
            file.write("Drone operational costs (L-type drones):\n")
            for i in range(len(problem_instance['facilities'])):
                for j in range(len(problem_instance['customers'])):
                    file.write(f"Facility {i} to Customer {j}: {problem_instance['cost_l'][i][j]}\n")
            file.write("Facility capacities:\n")
            for i in range(len(problem_instance['facilities'])):
                file.write(f"Facility {i}: {problem_instance['facility_capacities'][i]}\n")
            file.write(f"Small drone payload capacity: {problem_instance['payload_small']}\n")
            file.write(f"Large drone payload capacity: {problem_instance['payload_large']}\n")
            file.write(f"Small drone max range: {problem_instance['range_small']}\n")
            file.write(f"Large drone max range: {problem_instance['range_large']}\n")
        print(f"Problem instance saved to {folder_name}/{filename}.txt")
    except Exception as e:
        print(f"Error saving problem instance: {e}")

# Generate and save multiple problems with updated grid sizes
def generate_and_save_multiple_problems():
    for difficulty in range(1, num_classes + 1):
        print(f"Generating problems for difficulty class {difficulty}...")
        for problem_idx in range(problems_per_class):
            F = base_F + difficulty
            C = base_C + difficulty
            Ds = base_Ds + difficulty
            Dl = base_Dl + difficulty
            grid_size = grid_sizes[difficulty]  # Use the specified grid size for the class
            current_max_capacity = max_capacity + difficulty  # Avoid overriding global max_capacity
            current_max_payload_small = max_payload_small + difficulty
            current_max_payload_large = max_payload_large + difficulty * 2
            current_max_range_small = max_range_small + difficulty
            current_max_range_large = max_range_large + difficulty * 2

            problem_instance = generate_problem_instance(
                F, C, Ds, Dl, grid_size, current_max_capacity,
                current_max_payload_small, current_max_payload_large,
                current_max_range_small, current_max_range_large
            )
            filename = f"problem_class_{difficulty}_instance_{problem_idx + 1}"
            save_problem_to_disk(problem_instance, filename=filename)

# Example: Generate and save problems for all difficulty classes
generate_and_save_multiple_problems()
