import random
import numpy as np
import os

# Random Seed for Reproducibility
random.seed(69)
np.random.seed(69)

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

def generate_fixed_costs(n_facilities, fixed_cost_range):
    fixed_costs = np.random.randint(fixed_cost_range[0], fixed_cost_range[1] + 1, n_facilities)
    return fixed_costs

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
def generate_problem_instance(n_facilities, n_customers, grid_size, max_capacity, fixed_cost_range, var_cost_range_s, var_cost_range_l, max_payload_small, max_payload_large, max_range_small, max_range_large, Ds, Dl, package_weight_range):
    facilities = generate_points(n_facilities, grid_size)
    customers = generate_points(n_customers, grid_size)
    distances = compute_distances(facilities, customers)
    fixed_costs = generate_fixed_costs(n_facilities, fixed_cost_range)
    cost_s, cost_l = generate_costs(n_facilities, n_customers, var_cost_range_s, var_cost_range_l)
    facility_capacities = generate_facility_capacities(n_facilities, max_capacity)
    payload_small = random.randint(5, max_payload_small)
    payload_large = random.randint(10, max_payload_large)
    range_small = random.randint(10, max_range_small)
    range_large = random.randint(20, max_range_large)
    package_weights = generate_package_weights(n_customers, *package_weight_range)
    Ds = Ds
    Dl = Dl

    return {
        'facilities': facilities,
        'customers': customers,
        'distances': distances,
        'fixed_costs' : fixed_costs,
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
        'n_customers': n_customers,
        'Ds' : Ds,
        'Dl' : Dl,
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
        file.write("Facilities fixed costs:\n")
        for i in range(len(problem_instance['facilities'])):
            file.write(f"Facility {i}: {problem_instance['fixed_costs'][i]}\n")
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
        file.write(f"Number of S-type drones: {problem_instance['Ds']}\n")
        file.write(f"Number of L-type drones: {problem_instance['Dl']}\n")

# Generate and save multiple problem instances across classes
def generate_and_save_multiple_classes(num_problems_per_class=20):
    # Define the settings for each class using dictionaries for better clarity
    class_settings = [
        # Class 1 - OK
        {
            "grid_size": 10, "n_customers": 15, "n_facilities": 5, "max_capacity": 10,
            "fixed_cost_range": (10, 20), "var_cost_range_s": (1, 10), "var_cost_range_l" : (10, 30), 
            "max_payload_s": 15, "max_payload_l": 30, "max_range_s": 15, "max_range_l": 30, "Ds": 20, 
            "Dl": 20, "package_weight_range": (1, 30)
        },
        # Less than 1 sec solver time

        # Class 2 - OK
        {
            "grid_size": 15, "n_customers": 20, "n_facilities": 30, "max_capacity": 20,
            "fixed_cost_range": (10, 20), "var_cost_range_s": (1, 10), "var_cost_range_l" : (10, 20), 
            "max_payload_s": 10, "max_payload_l": 30, "max_range_s": 15, "max_range_l": 30, "Ds": 30, 
            "Dl": 20, "package_weight_range": (1, 20)
        },
        # ~ 5 sec solver time

        # Class 3 - OK
        {
            "grid_size": 20, "n_customers": 30, "n_facilities": 40, "max_capacity": 20,
            "fixed_cost_range": (10, 30), "var_cost_range_s": (5, 15), "var_cost_range_l" : (10, 25), 
            "max_payload_s": 15, "max_payload_l": 30, "max_range_s": 20, "max_range_l": 30, "Ds": 20, 
            "Dl": 20, "package_weight_range": (5, 30)
        },
        # ~ 10 sec solver time

        # Class 4 - OK
        {
            "grid_size": 40, "n_customers": 53, "n_facilities": 40, "max_capacity": 40,
            "fixed_cost_range": (1, 50), "var_cost_range_s": (1, 15), "var_cost_range_l" : (15, 25), 
            "max_payload_s": 30, "max_payload_l": 70, "max_range_s": 30, "max_range_l": 60, "Ds": 35, 
            "Dl": 45, "package_weight_range": (5, 50)
        },
        # ~ 20+ - 35 sec solver time

        # Class 5 - OK
        {
            "grid_size": 80, "n_customers": 25, "n_facilities": 20, "max_capacity": 60,
            "fixed_cost_range": (1, 50), "var_cost_range_s": (1, 15), "var_cost_range_l" : (15, 25), 
            "max_payload_s": 30, "max_payload_l": 70, "max_range_s": 60, "max_range_l": 120, "Ds": 200, 
            "Dl": 200, "package_weight_range": (5, 50)
        },
        # ~ 1 min solver time

        # Class 6 - OK
        {
            "grid_size": 160, "n_customers": 70, "n_facilities": 40, "max_capacity": 80,
            "fixed_cost_range": (1, 40), "var_cost_range_s": (5, 15), "var_cost_range_l" : (10, 25), 
            "max_payload_s": 25, "max_payload_l": 60, "max_range_s": 130, "max_range_l": 220, "Ds": 150, 
            "Dl": 150, "package_weight_range": (5, 30)
        },
        # ~ 5 min solver time
         
        # Class 7 - Testing Now
        {
            "grid_size": 200, "n_customers": 100, "n_facilities": 50, "max_capacity": 100,
            "fixed_cost_range": (1, 50), "var_cost_range_s": (5, 15), "var_cost_range_l": (10, 25), 
            "max_payload_s": 25, "max_payload_l": 60, "max_range_s": 150, "max_range_l": 250, "Ds": 200, 
            "Dl": 200, "package_weight_range": (5, 40)
        },
        # ~ 10 min solver time

        # Class 8 - Testing Now
        {
            "grid_size": 250, "n_customers": 130, "n_facilities": 70, "max_capacity": 120,
            "fixed_cost_range": (1, 50), "var_cost_range_s": (5, 20), "var_cost_range_l": (10, 30), 
            "max_payload_s": 30, "max_payload_l": 70, "max_range_s": 180, "max_range_l": 300, "Ds": 250, 
            "Dl": 250, "package_weight_range": (5, 50)
        },
        # ~ 15 min solver time
        
        # Class 9 - Testing Now
        {
            "grid_size": 300, "n_customers": 160, "n_facilities": 80, "max_capacity": 150,
            "fixed_cost_range": (1, 60), "var_cost_range_s": (5, 20), "var_cost_range_l": (10, 30), 
            "max_payload_s": 35, "max_payload_l": 80, "max_range_s": 200, "max_range_l": 350, "Ds": 300, 
            "Dl": 300, "package_weight_range": (10, 60)
        },
        # ~ 20 min solver time

        # Class 10 - Testing Now
        {
            "grid_size": 400, "n_customers": 200, "n_facilities": 100, "max_capacity": 200,
            "fixed_cost_range": (1, 70), "var_cost_range_s": (10, 25), "var_cost_range_l": (15, 35), 
            "max_payload_s": 40, "max_payload_l": 90, "max_range_s": 250, "max_range_l": 400, "Ds": 400, 
            "Dl": 400, "package_weight_range": (10, 70)
        }
        # ~ 30 min solver time
    ]

    # Iterate over the class settings and generate problem instances
    for class_num, setting in enumerate(class_settings, start=1):
        for problem_num in range(1, num_problems_per_class + 1):
            problem_instance = generate_problem_instance(
                setting["n_facilities"], setting["n_customers"], setting["grid_size"], setting["max_capacity"],
                setting["fixed_cost_range"], setting["var_cost_range_s"], setting["var_cost_range_l"], setting["max_payload_s"], setting["max_payload_l"],
                setting["max_range_s"], setting["max_range_l"], setting["Ds"], setting["Dl"], setting["fixed_cost_range"]
            )
            save_problem_to_disk(problem_instance, class_num, problem_num)

generate_and_save_multiple_classes()
