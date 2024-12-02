# Facility Location Optimization for Dual-Capacity Drone Fleet in Last-Mile Delivery Operations
## Problem Description:

The goal is to determine the optimal locations for facilities (e.g., drone hubs, recharging stations) to serve customer demand points effectively. Additionally, this problem involves choosing the appropriate type of drone for each delivery, balancing between **small-capacity drones** (for lighter, shorter deliveries) and **large-capacity drones** (for heavier loads or longer distances).

## Objective:

Minimize the total cost, which includes:

1. **Facility setup and operational costs.**
2. **Travel and operational costs** for each type of drone.
3. **Drone selection costs** based on the load and distance requirements for each customer.

## Constraints:

1. **Facility Assignment Constraint**: Each customer must be served by one facility and one type of drone.
2. **Facility Capacity Constraint**: Each facility has a total capacity limit for the number of drones it can support, with each type of drone occupying a different amount of space.
3. **Assignment Activation Constraint**: A customer can only be assigned to a facility that is open.
4. **PayloadCapacityConstraint**:The assigned drone type must be able to carry the package weight.
5. **Distance/RangeConstraint**: Small-capacity drones can only serve customers within range $R_{s}$, and large capacity drones within range $R_{l}$.


## Parameters:
$𝐹$ : Potential facility locations **(Set)** <br>
$𝐶$ : Customer locations **(Set)** <br>
$D_{𝑠}$ : Small capacity drones **(Set)** <br>
$𝐷_{𝑙}$ : Large capacity drones **(Set)** <br>
$d_{𝑖𝑗}$ : Distance between facility i and customer j, $𝑖\in𝐹$ & $𝑗\in𝐶$ **(2D Array)** <br>
$𝑓_{𝑖}$ : Fixed cost of operating facility i, $𝑖\in𝐹$ **(Array)** <br>
$𝑐_{𝑖𝑗}^s$ : Operational cost for serving customer j from facility i with an S-type drone **(2D Array)** <br>
$𝑐_{𝑖j}^l$ : Operational cost for serving customer j from facility i with an L-type drone **(2D Array)** <br>
$𝐾_{𝑖}$ : The capacity of each facility, the maximum number of drones it can support **(Array)** <br>
$𝑊_{𝑗}$ : Weight of customer’s j package **(Array)** <br>
$𝑤_{𝑠}$ : Weight factor for S-type drones for the facilities **(Constant)** <br>
$𝑤_{𝑙}$ : Weight factor for L-type drones for the facilities **(Constant)** <br>
$𝑃_{𝑠}$ : Payload capacity of S-type drones **(Constant)** <br>
$𝑃_{𝑙}$ : Payload capacity of L-type drones **(Constant)** <br>
$𝑅_{𝑠}$ : Max Range of S-type drones **(Constant)** <br>
$𝑅_{𝑙}$ : Max Range of L-type drones **(Constant)** <br>

## Decision Variables:
$x_{i} \in {0,1}$: Binary variable. Facility i is opened (1) or closed (0) <br>
$y_{ij}^s \in {0,1}$: Binary variable. If customer j is served by facility i using an S-type drone (1), otherwise (0) <br>
$y_{ij}^l \in {0,1}$: Binary variable. If customer j is served by facility i usingan L-type drone (1), otherwise (0) <br>


## ObjectiveFunction:
Minimize total cost. That includes the fixed costs and variable costs. <br>

$\displaystyle\sum_{i \in F} 𝑓_{𝑖} ∙ x_{i} + \displaystyle\sum_{i \in F} \displaystyle\sum_{j \in C} (𝑐_{𝑖𝑗}^s ∙ y_{ij}^s + 𝑐_{𝑖𝑗}^l ∙ y_{ij}^l)$ 

## Constraints:
### 1. Each customer must be served by one facility and one type of drone.
$\displaystyle\sum_{i \in D_{s}} \displaystyle\sum_{i \in F}  y_{ij}^s + \displaystyle\sum_{i \in D_{l}} \displaystyle\sum_{i \in F}  y_{ij}^l = 1, \quad \forall j \in C$

### 2. Each facility has  a total capacity limit for the number of drones it can support.
$\displaystyle\sum_{j \in C} (𝑤_{𝑠} ∙ y_{ij}^s + 𝑤_{l} ∙ y_{ij}^l) \leq K_{i} ∙ x_{i}, \quad \forall i \in F$
 
### 3. A customer can only be assigned to a facility that is open.
$y_{ij}^s \leq x_{i}$ & $y_{ij}^l \leq x_{i}, \quad \forall i \in F, \quad \forall j \in C$

### 4. The assigned drone type must be able to carry the package weight.
$W_{j} ∙ y_{ij}^s \leq P_{s}$ & $W_{j} ∙ y_{ij}^l \leq P_{l}, \quad \forall i \in F, \quad \forall j \in C$

### 5. S-type drones can only serve customers within $R_{s}$ while L-type drones can serve customers within $R_{l}$
$d_{ij} ∙ y_{ij}^s \leq R_{s}$ & $d_{ij} ∙ y_{ij}^l \leq R_{l}, \quad \forall i \in F, \quad \forall j \in C$

### 6. Each drone (small or large) should be assigned to exactly one customer


### 7. Each drone should be assigned to exactly one facility.


