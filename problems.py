import numpy as np
import gurobipy as gp
from gurobipy import GRB

# Create a function that reads data from a file
def read_data(filename):
    f = open(filename, "rt")
    lines = f.readlines()
    lines = [f.replace('\n', '') for f in lines if f != '\n']
    line = lines[0].split(' ')
    P = int(line[3])
    N = int(line[2])
    CL = int(line[1])
    line_idx = 2
    clients = np.zeros(CL)
    for i in range(CL):
        clients[i] = int(lines[line_idx])
        line_idx+=1

    line_idx+=1
    line = lines[line_idx]
    points = np.zeros(N)
    for i in range(N):
        points[i] = int(lines[line_idx])
        line_idx+=1
    
    line_idx+=1
    unaryConstraint = np.zeros(P)
    for i in range(P):
        unaryConstraint[i] = lines[line_idx].split(' ')[1]
        line_idx+=1
    
    line_idx+=1
    line = lines[line_idx]
    binaryConstraint = np.zeros(shape=(P,P))
    for i in range(P-1):
        for j in range(i+1, P):
            line = lines[line_idx].split(' ')
            binaryConstraint[i][j] = float(line[2])
            binaryConstraint[j][i] = float(line[2])
            line_idx+=1

    line_idx+=1
    line = lines[line_idx]
    fp_euc_distances = np.zeros(shape=(N,N))
    fp_sp_distances = np.zeros(shape=(N,N))
    for i in range(N):
        for j in range(N):
            if(i==j):
                continue
            line = lines[line_idx].split(' ')
            fp_sp_distances[i][j] = float(line[2])
            fp_euc_distances[i][j] = float(line[3])
            line_idx+=1

    line_idx+=1
    line = lines[line_idx]
    cl_euc_distances = np.zeros(shape=(CL,N))
    cl_sp_distances = np.zeros(shape=(CL,N))
    for i in range(CL):
            for j in range(N):
                line = lines[line_idx].split(' ')
                cl_sp_distances[i][j] = float(line[2])
                cl_euc_distances[i][j] = float(line[3])
                line_idx+=1


    return P,N,CL,clients,points,unaryConstraint,binaryConstraint,fp_sp_distances,fp_euc_distances,cl_euc_distances,cl_sp_distances

# Creates a pcenter problem and returns A, b and c matrices
def pcenter_Matrix(filename):
    
    P,N,CL,clients,points,unaryConstraint,binaryConstraint,fp_sp_distances,fpEucDistances,clEucDistances,clSpDistances = read_data(filename)
    
    vars = N * P + CL * N + 1
    A = list()
    A_tmp = np.zeros(vars, dtype=int)
    A_tmp2 = np.zeros(vars, dtype=int)
    b = list()

    #linear constraint #1 : one facility site can host at most one facility
    for i in range(N):
        for j in range(P):
            A_tmp[i*P + j] = 1
        b.append(1)
        A.append(A_tmp)
        A_tmp = np.zeros(vars, dtype=int)

    """
    for i in A:
        print(i) 
    print(b)
    """
    #linear constraint #2 : one facility should be hosted at exactly once
    for i in range(P):
        for j in range(N):
            A_tmp[j*P + i] = 1
            A_tmp2[j*P + i] = -1
        b.append(1)
        A.append(A_tmp)
        A_tmp = np.zeros(vars, dtype=int)

        # equality constraint broke down to two inequalities. (  Cx+Dy=γ   ==>   Cx+Dy≤γ  and −Cx−Dy≤−γ  )
        A.append(A_tmp2)
        b.append(-1)
        A_tmp2 = np.zeros(vars, dtype=int)

    """
    for i in A:
        print(i) 
    print(b)
    """
    #linear constraint #3 : p facilities should be opened
    for i in range(N):
        for j in range(P):
            A_tmp[i*P + j] = 1
            A_tmp2[i*P + j] = -1
    b.append(P)
    A.append(A_tmp)
    A_tmp = np.zeros(vars, dtype=int)

    # equality constraint broke down to two inequalities. (  Cx+Dy=γ   ==>   Cx+Dy≤γ  and −Cx−Dy≤−γ  )
    A.append(A_tmp2)
    b.append(-P)
    A_tmp2 = np.zeros(vars, dtype=int)

    """
    for i in A:
        print(i) 
    print(b)
    """

    #linear constraint #4 : each demand node will be served by one facility location.
    for i in range(CL):
        for j in range(N):
            A_tmp[P*N + i*N + j] = 1
            A_tmp2[P*N + i*N + j] = -1
        b.append(1)
        A.append(A_tmp)
        A_tmp = np.zeros(vars, dtype=int)

        # equality constraint broke down to two inequalities. (  Cx+Dy=γ   ==>   Cx+Dy≤γ  and −Cx−Dy≤−γ  )
        A.append(A_tmp2)
        b.append(-1)
        A_tmp2 = np.zeros(vars, dtype=int)

    """
    for i in A:
        print(i) 
    print(b)
    """
    #linear constraint #5 : each demand node will be served by a location that is opened (Balinski constraints).
    expr = 0
    for i in range(N):
        for j in range(CL):
            A_tmp[P*N + j*N + i] = 1
            for k in range(P):
                A_tmp[i*P + k] = -1
            b.append(0)
            A.append(A_tmp)
            A_tmp = np.zeros(vars, dtype=int)


    """
    for i in A:
        print(i) 
    print(b)
    """ 
    #linear constraint #6 : z should be larger than the distance from any client to the assigned facility.
    for i in range(CL):
        for j in range(N):
            A_tmp[P*N + i*N + j] = clSpDistances[i][j] 
        A_tmp[-1] = -1
        b.append(0)
        A.append(A_tmp)
        A_tmp = np.zeros(vars, dtype=int)


    """
    for i in A:
        print(i) 
    print(b)
    """
    #linear constraint #7 : each demand node should be at a safe distance from each facility site if it is opened.
    for i in range(CL):
        for j in range(N):
            for p in range(P):
                if clEucDistances[i][j] <= unaryConstraint[p]:
                    A_tmp[j*P + p] = 1
                    b.append(0)
                    A.append(A_tmp)
                    A_tmp = np.zeros(vars, dtype=int)

    """
    for i in A:
        print(i) 
    print(b)
    """
    #linear constraint #8 : each facility should be at a safe distance from other facilities.
    for i in range(N):
        for j in range(N):
            if i==j:
                continue
            for p in range(P-1):
                for l in range(p+1, P):
                    if fpEucDistances[i][j] <= binaryConstraint[p][l]:
                        A_tmp[i*P + p] = 1
                        A_tmp[j*P + l] = 1
                        b.append(1)
                        A.append(A_tmp)
                        A_tmp = np.zeros(vars, dtype=int)

    """
    for i in A:
        print(i) 
    print(b)
    """
    A = np.array(A)
    b= np.array(b)
    c = np.zeros(vars, dtype=int)
    c[-1] = 1


    # Select the integer var
    integer_var = [True for i in range(vars)]
    integer_var[-1] = False

    # Lower bound for vars 
    lb = [0 for i in range(vars)]
    # Upper bound for vars
    ub = [1 for i in range(vars)]

    lb[-1] = -GRB.INFINITY
    ub[-1] = GRB.INFINITY
    

    return c, A, b, integer_var, lb, ub

# Creates a model for the pcenter problem for gurobipy
def pcenter(filename):
    P,N,CL,clients,points,unaryConstraint,binaryConstraint,fp_sp_distances,fpEucDistances,clEucDistances,clSpDistances = read_data(filename)

    num_vars = N*P + CL*N + 1
    model = gp.Model()

    ub = [1 if i < num_vars-1 else np.inf for i in range(num_vars) ]
    lb = [0 if i < num_vars-1 else -np.inf for i in range(num_vars) ]


    # Create variables
    x = model.addVars(num_vars, lb=lb, ub=ub, vtype=GRB.CONTINUOUS, name="x")

    #linear constraint #1 : one facility site can host at most one facility
    expr = 0
    for i in range(N):
        for j in range(P):
            expr += x[i*P + j]
        model.addLConstr(expr <= 1)
        expr = 0

    #linear constraint #2 : one facility should be hosted at exactly once
    expr = 0
    for i in range(P):
        for j in range(N):
            expr += x[j*P + i]
        model.addLConstr(expr == 1)
        expr = 0

    #linear constraint #3 : p facilities should be opened
    expr = 0
    for i in range(N):
        for j in range(P):
            expr += x[i*P + j]
    model.addLConstr(expr == P)


    #linear constraint #4 : each demand node will be served by one facility location.
    expr = 0
    for i in range(CL):
        for j in range(N):
            expr += x[N*P + i*N + j]
        model.addLConstr(expr == 1)
        expr = 0


    #linear constraint #5 : each demand node will be served by a location that is opened (Balinski constraints).
    expr = 0
    for i in range(N):
        for j in range(CL):
            expr += x[N*P + j*N + i]
            for k in range(P):
                expr += -x[i*P + k]
            model.addLConstr(expr <= 0)
            expr = 0

    #linear constraint #6 : z should be larger than the distance from any client to the assigned facility.
    expr = 0
    for i in range(CL):
        for j in range(N):
            expr += clSpDistances[i][j] * x[N*P + i*N + j]
        expr += -x[N*P + CL*N]
        model.addLConstr(expr <= 0)
        expr = 0

    #linear constraint #7 : each demand node should be at a safe distance from each facility site if it is opened.
    expr = 0
    for i in range(CL):
        for j in range(N):
            for p in range(P):
                if clEucDistances[i][j] <= unaryConstraint[p]:
                    model.setAttr("UB", x[j*P + p], 0)
                    ub[j*P + p] = 0

    #linear constraint #8 : each facility should be at a safe distance from other facilities.
    expr = 0
    for i in range(N):
        for j in range(N):
            if i==j:
                continue
            for p in range(P-1):
                for l in range(p+1, P):
                    if fpEucDistances[i][j] <= binaryConstraint[p][l]:
                        model.addLConstr(x[i*P + p] + x[j*P +l] <= 1)

    c = [0 if i < num_vars-1 else 1 for i in range(num_vars)]
    
    model.setObjective(gp.quicksum(c[i] * x[i] for i in range(num_vars)))
    model.ModelSense = GRB.MINIMIZE # We have minimization
    model.Params.method = 1  # 1 indicates the dual Simplex algorithm in Gurobi
    model.update()

    # Save the model file
    #model.write("model.lp")

    # Define which variables should have integer values
    integer_var = [True if i < num_vars-1 else False for i in range(num_vars)]

    return model, ub, lb, integer_var, num_vars, c
    

def very_small_random_problem():

    # Objective Function Coefficient
    c = np.array([4, 3, 1])
    # Constraint
    A = np.array([[3, 2, 1], [2, 1, 2]])
    b = np.array([7, 11])
    # Select the integer var
    integer_var = [False, True, True]
    # Lower bound for vars 
    lb = [0, 0, 0]
    # Upper bound for vars 
    ub = [GRB.INFINITY, GRB.INFINITY, GRB.INFINITY]

    return c, A, b, integer_var, lb, ub

def very_simple_random_problem():
    # Objective Function Coefficient
    c = np.array([-8, -2, -4, -7, -5])
    # Constraint
    A = np.array([[-3, -3, 1, 2, 3], [-5, -3, -2, -1, 1]])
    b = np.array([-2, -4])
    # Select the integer var
    integer_var = [True, True, True, True, True]
    # Lower bound for vars 
    lb = [0, 0, 0, 0, 0]
    ub = [1, 1, 1, 1, 1]

    return c, A, b, integer_var, lb, ub