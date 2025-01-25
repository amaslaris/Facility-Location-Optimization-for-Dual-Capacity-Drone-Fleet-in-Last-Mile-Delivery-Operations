from gurobipy import Model, GRB
from collections import deque
import numpy as np

# Reading problem instance from file 
def read_problem_instance(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Data Types
    # Parse data
    num_facilities = int(lines[1].split(': ')[1])
    num_customers = int(lines[2].split(': ')[1])
    # grid_size = int(lines[3].split(': ')[1].split('x')[0])

    
    facilities_coords = []
    customers_coords = []
    distances = np.zeros((num_facilities, num_customers))
    fixed_costs = np.zeros((num_facilities), dtype=int)
    facility_capacities = []
    customer_weights = []  
    cost_s = np.zeros((num_facilities, num_customers))  
    cost_l = np.zeros((num_facilities, num_customers))  

    line_idx = 5  

    # Facilities coordinates
    for _ in range(num_facilities):
        x, y = map(int, lines[line_idx].split(','))
        facilities_coords.append((x, y))
        line_idx += 1

    # Customers coordinates
    line_idx += 1
    for _ in range(num_customers):
        x, y = map(int, lines[line_idx].split(','))
        customers_coords.append((x, y))
        line_idx += 1

    # Distances
    line_idx += 1
    for i in range(num_facilities):
        for j in range(num_customers):
            distances[i][j] = float(lines[line_idx].split(': ')[1])
            line_idx += 1

    # Fixed costs for facility openings
    line_idx += 1
    for i in range(num_facilities):
        fixed_costs[i] = int(lines[line_idx].split(': ')[1])
        line_idx += 1

    # Drone operational costs (small drone)
    line_idx += 1
    for i in range(num_facilities):
        for j in range(num_customers):
            cost_s[i][j] = float(lines[line_idx].split(': ')[1])
            line_idx += 1

    # Drone operational costs (large drone)
    line_idx += 1
    for i in range(num_facilities):
        for j in range(num_customers):
            cost_l[i][j] = float(lines[line_idx].split(': ')[1])
            line_idx += 1

    # Facility capacities
    line_idx += 1
    for _ in range(num_facilities):
        facility_capacities.append(int(lines[line_idx].split(': ')[1]))
        line_idx += 1

    # Small and large drone capacities and ranges
    payload_small = int(lines[line_idx].split(': ')[1])
    payload_large = int(lines[line_idx + 1].split(': ')[1])
    range_small = int(lines[line_idx + 2].split(': ')[1])
    range_large = int(lines[line_idx + 3].split(': ')[1])

    # Check if customer weights section exists
    if line_idx + 4 < len(lines) and lines[line_idx + 4].startswith("Package weights"):
        # Read customer weights if the section exists
        line_idx += 5  # Skip to the section after the drone capacities and ranges
        for _ in range(num_customers):
            customer_weight = int(lines[line_idx].split(': ')[1])
            customer_weights.append(customer_weight)
            line_idx += 1
    else:
        # If no package weights section, assign default weights or handle error
        print("Warning: No customer weights found. Using default weights.")
        customer_weights = [1] * num_customers  # Assign default weight of 1

    Ds = int(lines[line_idx].split(': ')[1])
    Dl = int(lines[line_idx + 1].split(': ')[1])

    return {
        "num_facilities": num_facilities,
        "num_customers": num_customers,
        "distances": distances,
        "fixed_costs" : fixed_costs,
        "facility_capacities": facility_capacities,
        "payload_small": payload_small,
        "payload_large": payload_large,
        "range_small": range_small,
        "range_large": range_large,
        "customer_weights": customer_weights,
        "cost_s": cost_s,
        "cost_l": cost_l,
        "Ds" : Ds,
        "Dl" : Dl
    }

# Model initialization and solve
# Initialize the model
def model_creation(file_path):
    model = Model()
    data = read_problem_instance(file_path)

    F = range(data['num_facilities'])
    C = range(data['num_customers'])
    Ds = range(data["Ds"])
    Dl = range(data["Dl"])

    d = data['distances']
    f = data['fixed_costs']
    cs = data['cost_s']
    cl = data['cost_l']
    K = data['facility_capacities']
    W = data['customer_weights']
    Ps = data['payload_small']
    Pl = data['payload_large']
    Rs = data['range_small']
    Rl = data['range_large']
    
    ws = 1
    wl = 2

    # Decision Variables 
    x = model.addVars(F, vtype=GRB.BINARY, name="x")
    ys = model.addVars(Ds, F, C, vtype=GRB.BINARY, name="ys")
    yl = model.addVars(Dl, F, C, vtype=GRB.BINARY, name="yl")

    model.update()

    # Objective Function
    penalty_large_drone = 0.5
    fixed_costs = sum(f[i] * x[i] for i in F)
    variable_costs = sum(cs[i][j] * ys[s, i, j] + (cl[i][j] + penalty_large_drone) * yl[l, i, j] for s in Ds for l in Dl for i in F for j in C)
    model.setObjective(fixed_costs + variable_costs, GRB.MINIMIZE)

    # Constraints
    for j in C:
        model.addConstr(sum(ys[s, i, j] for s in Ds for i in F) + sum(yl[l, i, j] for l in Dl for i in F) == 1)

    for i in F:
        model.addConstr(sum(ws * ys[s, i, j] for s in Ds for j in C) + sum(wl * yl[l, i, j] for l in Dl for j in C) <= K[i] * x[i])

    for s in Ds:
        for i in F:
            for j in C:
                model.addConstr(ys[s, i, j] <= x[i])

    for l in Dl:
        for i in F:
            for j in C:
                model.addConstr(yl[l, i, j] <= x[i])

    for s in Ds:
        for i in F:
            for j in C:
                model.addConstr(W[j] * ys[s, i, j] <= Ps)

    for l in Dl:
        for i in F:
            for j in C:
                model.addConstr(W[j] * yl[l, i, j] <= Pl)

    for s in Ds:
        for i in F:
            for j in C:
                model.addConstr(d[i][j] * ys[s, i, j] <= Rs)

    for l in Dl:
        for i in F:
            for j in C:
                model.addConstr(d[i][j] * yl[l, i, j] <= Rl)

    for s in Ds:
        model.addConstr(sum(ys[s, i, j] for i in F for j in C) <= 1)

    for l in Dl:
        model.addConstr(sum(yl[l, i, j] for i in F for j in C) <= 1)

    return model

# Calculate If a Value is Very Close to an Integer Value
def is_nearly_integer(value, tolerance=1e-6):
    return abs(value - round(value)) <= tolerance

# A Class 'Node' That Holds the Information of a Node
class Node:
    def __init__(self, ub, lb, depth, vbasis, cbasis, branching_var, label=""):
        self.ub = ub
        self.lb = lb
        self.depth = depth
        self.vbasis = vbasis
        self.cbasis = cbasis
        self.branching_var = branching_var
        self.label = label

# Print Debugging Info
def debug_print(node: Node = None, x_obj=None, sol_status=None):
    print("\n\n-----------------  DEBUG OUTPUT  -----------------\n\n")
    print(f"UB:{upper_bound}")
    print(f"LB:{lower_bound}")
    if node is not None:
        print(f"Branching Var: {node.branching_var}")
    if node is not None:
        print(f"Child: {node.label}")
    if node is not None:
        print(f"Depth: {node.depth}")
    if x_obj is not None:
        print(f"Simplex Objective: {x_obj}")
    if sol_status is not None:
        print(f"Solution status: {sol_status}")

    print("\n\n--------------------------------------------------\n\n")

# Branch & Bound Algorithm
def branch_and_bound(model, ub, lb, integer_var, best_bound_per_depth, nodes_per_depth, vbasis=[], cbasis=[], depth=0):
    global nodes, lower_bound, upper_bound

    # create stack using deque() structure
    stack = deque()

    # initialize solution list
    solutions = list()
    solutions_found = 0
    best_sol_idx = 0

    # initialize best solution
    if isMax:
        best_sol_obj = -np.inf
    else:
        best_sol_obj = np.inf

    # create root node
    root_node = Node(ub, lb, depth, vbasis, cbasis, -1, "root")
    nodes_per_depth[0] -= 1

    # ===============  Root node  ==========================
    if DEBUG_MODE:
        debug_print()

    # solve relaxed problem
    model.optimize()

    # check if the model was solved to optimality. If not then return (infeasible).
    if model.status != GRB.OPTIMAL:
        if isMax:
            if DEBUG_MODE:
                debug_print(node=root_node, sol_status="Infeasible")
            return [], -np.inf, depth
        else:
            if DEBUG_MODE:
                debug_print(node=root_node, sol_status="Infeasible")
            return [], np.inf, depth

    # get the solution (variable assignments)
    x_candidate = model.getAttr('X', model.getVars())

    # get the objective value
    x_obj = model.ObjVal
    best_bound_per_depth[0] = x_obj

    # check if all variables have integer values (from the ones that are supposed to be integers).
    # If not, then select the first variable with a fractional value to be the one fixed
    vars_have_integer_vals = True
    for idx, is_int_var in enumerate(integer_var):
        if is_int_var and not is_nearly_integer(x_candidate[idx]):
            vars_have_integer_vals = False
            selected_var_idx = idx
            break

    # found feasible solution
    if vars_have_integer_vals:
        # if we have a feasible solution in root, then terminate
        solutions.append([x_candidate, x_obj, depth])
        solutions_found += 1

        if DEBUG_MODE:
            debug_print(node=root_node, x_obj=x_obj, sol_status="Integer")
        return solutions, best_sol_idx, solutions_found

    # otherwise update lower/upper bound for min/max respectively
    else:
        if isMax:
            upper_bound = x_obj
        else:
            lower_bound = x_obj

    if DEBUG_MODE:
        debug_print(node=root_node, x_obj=x_obj, sol_status="Fractional")

    # Retrieve vbasis and cbasis
    vbasis = model.getAttr("VBasis", model.getVars())
    cbasis = model.getAttr("CBasis", model.getConstrs())

    # create lower bounds and upper bounds for the variables of the child nodes
    left_lb = np.copy(lb)
    left_ub = np.copy(ub)
    right_lb = np.copy(lb)
    right_ub = np.copy(ub)

    # create left and right branches (e.g. set left: x = 0, right: x = 1 in a binary problem)
    left_ub[selected_var_idx] = np.floor(x_candidate[selected_var_idx])
    right_lb[selected_var_idx] = np.ceil(x_candidate[selected_var_idx])

    # create child nodes
    left_child = Node(left_ub, left_lb, root_node.depth + 1, vbasis.copy(), cbasis.copy(), selected_var_idx, "Left")
    right_child = Node(right_ub, right_lb, root_node.depth + 1, vbasis.copy(), cbasis.copy(), selected_var_idx, "Right")

    # add child nodes in stack
    stack.append(right_child)
    stack.append(left_child)

    # solving sub problems
    # While the stack has nodes, continue solving
    while (len(stack) != 0):
        print("\n********************************  NEW NODE BEING EXPLORED  ******************************** ")

        # increment total nodes by 1
        nodes += 1

        # get the child node on top of stack
        current_node = stack[-1]

        # remove this node from stack
        stack.pop()

        # increase the nodes visited for current depth
        nodes_per_depth[current_node.depth] -= 1

        # warm start solver. Use the vbasis and cbasis that parent node passed to the current one.
        if (len(current_node.vbasis) != 0) and (len(current_node.cbasis) != 0):
            model.setAttr("VBasis", model.getVars(), current_node.vbasis)
            model.setAttr("CBasis", model.getConstrs(), current_node.cbasis)

        # update the state of the model, passing the new lower bounds/upper bounds for the vars.
        # Basically, we only change the ub/lb for the branching variable. Another way is to introduce a new constraint (e.g. x_i <= ub).
        model.setAttr("LB", model.getVars(), current_node.lb)
        model.setAttr("UB", model.getVars(), current_node.ub)
        model.update()

        if DEBUG_MODE:
            debug_print()

        # optimize the model
        model.optimize()

        # Check if the model was solved to optimality. If not then do not create child nodes.
        infeasible = False
        if model.status != GRB.OPTIMAL:
            if isMax:
                infeasible = True
                x_obj = -np.inf
            else:
                infeasible = True
                x_obj = np.inf
            for i in range(current_node.depth + 1, len(nodes_per_depth)):
                nodes_per_depth[i] -= 2 * (i - current_node.depth)

        else:
            # get the solution (variable assignments)
            x_candidate = model.getAttr('X', model.getVars())

            # get the objective value
            x_obj = model.ObjVal

            # update best bound per depth if a better solution was found
            if isMax == True and x_obj > best_bound_per_depth[current_node.depth]:
                best_bound_per_depth[current_node.depth] = x_obj
            elif isMax == False and x_obj < best_bound_per_depth[current_node.depth]:
                best_bound_per_depth[current_node.depth] = x_obj

            # if we reached the final node of a depth, then update the bounds
            if nodes_per_depth[current_node.depth] == 0:
                if isMax == True:
                    upper_bound = best_bound_per_depth[current_node.depth]
                else:
                    lower_bound = best_bound_per_depth[current_node.depth]

        # if infeasible don't create children (continue searching the next node)
        if infeasible:
            if DEBUG_MODE:
                debug_print(node=current_node, sol_status="Infeasible")
            continue

        # check if all variables have integer values (from the ones that are supposed to be integers)
        vars_have_integer_vals = True
        for idx, is_int_var in enumerate(integer_var):
            if is_int_var and not is_nearly_integer(x_candidate[idx]):
                vars_have_integer_vals = False
                selected_var_idx = idx
                break

        # found feasible solution
        if vars_have_integer_vals: # integer solution
            if isMax:
                if lower_bound < x_obj: # a better solution was found
                    lower_bound = x_obj
                    if abs(lower_bound - upper_bound) < 1e-6: # optimal solution
                        # store solution, number of solutions and best sol index (and return)
                        solutions.append([x_candidate, x_obj, current_node.depth])
                        solutions_found += 1
                        if (abs(x_obj - best_sol_obj) < 1e-6) or solutions_found == 1:
                            best_sol_obj = x_obj
                            best_sol_idx = solutions_found - 1

                            if DEBUG_MODE:
                                debug_print(node=current_node, x_obj=x_obj, sol_status="Integer/Optimal")
                        return solutions, best_sol_idx, solutions_found

                    # Not optimal. Store solution, number of solutions and best sol index (and do not expand children)
                    solutions.append([x_candidate, x_obj, current_node.depth])
                    solutions_found += 1
                    if (abs(x_obj - best_sol_obj) <= 1e-6) or solutions_found == 1:
                        best_sol_obj = x_obj
                        best_sol_idx = solutions_found - 1

                    # remove the children nodes from each next depth
                    for i in range(current_node.depth + 1, len(nodes_per_depth)):
                        nodes_per_depth[i] -= 2 * (i - current_node.depth)

                    if DEBUG_MODE:
                        debug_print(node=current_node, x_obj=x_obj, sol_status="Integer")
                    continue

            else:
                if upper_bound > x_obj: # better solution
                    upper_bound = x_obj # update bound
                    if abs(lower_bound - upper_bound) < 1e-6: # optimality reached
                        # store solution, number of solutions and best sol index (and return)
                        solutions.append([x_candidate, x_obj, current_node.depth])
                        solutions_found += 1
                        if (abs(x_obj - best_sol_obj) <= 1e-6) or solutions_found == 1:
                            best_sol_obj = x_obj
                            best_sol_idx = solutions_found - 1
                            if DEBUG_MODE:
                                debug_print(node=current_node, x_obj=x_obj, sol_status="Integer/Optimal")
                        return solutions, best_sol_idx, solutions_found

                    # Not optimal. Store solution, number of solutions and best sol index (and do not expand children)
                    solutions.append([x_candidate, x_obj, current_node.depth])
                    solutions_found += 1
                    if (abs(x_obj - best_sol_obj) <= 1e-6) or solutions_found == 1:
                        best_sol_obj = x_obj
                        best_sol_idx = solutions_found - 1

                    # remove the children nodes from each next depth
                    for i in range(current_node.depth + 1, len(nodes_per_depth)):
                        nodes_per_depth[i] -= 2 * (i - current_node.depth)

                    if DEBUG_MODE:
                        debug_print(node=current_node, x_obj=x_obj, sol_status="Integer")
                    continue

            # do not branch further if is an equal solution
            # remove the children nodes from each next depth
            for i in range(current_node.depth + 1, len(nodes_per_depth)):
                nodes_per_depth[i] -= 2 * (i - current_node.depth)

            if DEBUG_MODE:
                debug_print(node=current_node, x_obj=x_obj,
                            sol_status="Integer (Rejected -- Doesn't improve incumbent)")
            continue

        if isMax:
            if x_obj < lower_bound or abs(x_obj - lower_bound) < 1e-6: # cut
                # remove the children nodes from each next depth
                for i in range(current_node.depth + 1, len(nodes_per_depth)):
                    nodes_per_depth[i] -= 2 * (i - current_node.depth)
                if DEBUG_MODE:
                    debug_print(node=current_node, x_obj=x_obj, sol_status="Fractional -- Cut by bound")
                continue
        else:
            if x_obj > upper_bound or abs(x_obj - upper_bound) < 1e-6: # cut
                # remove the children nodes from each next depth
                for i in range(current_node.depth + 1, len(nodes_per_depth)):
                    nodes_per_depth[i] -= 2 * (i - current_node.depth)
                if DEBUG_MODE:
                    debug_print(node=current_node, x_obj=x_obj, sol_status="Fractional -- Cut by bound")
                continue

        if DEBUG_MODE:
            debug_print(node=current_node, x_obj=x_obj, sol_status="Fractional")

        # Retrieve vbasis and cbasis
        vbasis = model.getAttr("VBasis", model.getVars())
        cbasis = model.getAttr("CBasis", model.getConstrs())

        # create lower bounds and upper bounds for child nodes
        left_lb = np.copy(current_node.lb)
        left_ub = np.copy(current_node.ub)
        right_lb = np.copy(current_node.lb)
        right_ub = np.copy(current_node.ub)

        # create left and right branches  (e.g. set left: x = 0, right: x = 1 in a binary problem)
        left_ub[selected_var_idx] = np.floor(x_candidate[selected_var_idx])
        right_lb[selected_var_idx] = np.ceil(x_candidate[selected_var_idx])

        # create child nodes
        left_child = Node(left_ub, left_lb, current_node.depth + 1, vbasis.copy(), cbasis.copy(), selected_var_idx,
                          "Left")
        right_child = Node(right_ub, right_lb, current_node.depth + 1, vbasis.copy(), cbasis.copy(), selected_var_idx,
                           "Right")

        # add child nodes in stack
        stack.append(right_child)
        stack.append(left_child)

    return solutions, best_sol_idx, solutions_found

if __name__ == "__main__":

    problem_file = f"generated_problems/class_{1}/problem_instance_{2}.txt"
    model = model_creation(problem_file)

    # Define bounds for variables
    lb = [0, 0]  # lower bounds for x and y
    ub = [GRB.INFINITY, GRB.INFINITY]  # upper bounds for x and y
    integer_var = [1, 1]  # both x and y are integers

    # Set global parameters
    nodes = 0
    lower_bound = -np.inf
    upper_bound = np.inf
    DEBUG_MODE = False  # optional: enables debug prints
    isMax = False       # maximization problem

    # Track bounds and nodes per depth
    best_bound_per_depth = [-np.inf] * 10
    nodes_per_depth = [2**i for i in range(10)]

    solutions, best_sol_idx, solutions_found = branch_and_bound(model= model, ub= ub, lb= lb, integer_var= integer_var, best_bound_per_depth= best_bound_per_depth,nodes_per_depth= nodes_per_depth)

    # Print results
    if solutions:
        print(f"Best solution found: {solutions[best_sol_idx]}")
        print(f"Number of solutions found: {solutions_found}")
    else:
        print("No feasible solution found.")