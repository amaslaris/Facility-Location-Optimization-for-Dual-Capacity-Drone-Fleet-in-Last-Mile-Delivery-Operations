**Facility Location Optimization for Dual-Capacity Drone Fleet in Last-Mile Delivery Operations**

**Problem Description:**

The goal is to determine the optimal locations for facilities (e.g., drone hubs, recharging stations) to serve customer demand points effectively. Additionally, this problem involves choosing the appropriate type of drone for each delivery, balancing between **small-capacity drones** (for lighter, shorter deliveries) and **large-capacity drones** (for heavier loads or longer distances).

**Objective:**

Minimize the total cost, which includes:

1. **Facility setup and operational costs.**
1. **Travel and operational costs** for each type of drone.
1. **Drone selection costs** based on the load and distance requirements for each customer.

**Constraints:**

1. **Facility Assignment Constraint**: Each customer must be served by one facility and one type of drone.
1. **Facility Capacity Constraint**: Each facility has a total capacity limit for the number of drones it can support, with each type of drone occupying a different amount of space.
1. **Assignment Activation Constraint**: A customer can only be assigned to a facility that is open.
1. **Payload Capacity Constraint**: The assigned drone type must be able to carry the package weight.
1. **Distance/Range Constraint**: Small-capacity drones can only serve customers within range Rs , and large-capacity drones within range Rl.

**Parameters:**

- **:** Potential facility locations # (Set)
- **:** Customer locations # (Set)
- **:** Small capacity drones # (Set)
- **:** Large capacity drones # (Set)
- **:** Distance between facility i and customer j (  ∈  ,  ∈  ) # (2d Array)
- **:** Fixed cost of operating facility i ( ∈  ) # (Array)
- **:** Operational cost for serving customer j from facility i with an **S**-type drone # (2d Array)
- **:** Operational cost for serving customer j from facility i with an **L**-type drone # (2d Array)
- **:** The capacity of each facility, the maximum number of drones it can

support # (Array)

- **:** Weight of customer’s j package # (Array)
- **:** Weight factor for **S**-type drones for the facilities # (Constant)
- **:** Weight factor for **L**-type drones for the facilities # (Constant)
- **:** Payload capacity of **S**-type drones # (Constant)
- **:** Payload capacity of **L**-type drones # (Constant)
- **:** Max Range of **S**-type drones (Constant)
- **:** Max Range of **L**-type drones (Constant)

**Decision Variables:**

- ∈ {0, 1} **:** Binary variable. Facility i is opened (1) or closed (0)
- ∈ {0, 1} : Binary variable. If customer j is served by facility i using an **S**-type drone (1), otherwise (0)
- ∈ {0, 1} **:** Binary variable. If customer j is served by facility i using an

**L**-type drone (1), otherwise (0)

**Objective Function:**

Minimize total cost. That includes the fixed costs and variable costs.

- · + ∑ ∑ ( ·  + · )

∈ ∈ ∈

**Constraints:**

1. **Each customer must be served by one facility and one type of drone.**
- (   +  ) = 1, ∀ ∈

  ∈

2. **Each facility has a total capacity limit for the number of drones it can support.**
- ( ·   +  · ) ≤ · , ∀  ∈ ∈
3. **A customer can only be assigned to a facility that is open.**

≤  &   ≤  ,  ∀ ∈ , ∀ ∈ 

4. **The assigned drone type must be able to carry the package weight.**
- ≤  &   · ≤  ,  ∀ ∈ , ∀ ∈ 
5. **S-type drones can only serve customers within while L-type drones can serve customers within**
- ≤  &   · ≤  ,  ∀ ∈ , ∀ ∈ 

Σχετική Βιβλιογραφία:

1. [**Optimizing drone**-assisted **last**-**mile** deliveries: The vehicle routing problem with flexible **drones**](https://optimization-online.org/wp-content/uploads/2020/04/7737.pdf)
1. [Last-Mile Drone Delivery: Past, Present, and Future](https://www.mdpi.com/2504-446X/7/2/77)
1. [Optimal drone deployment for cost-effective and sustainable last-mile delivery operations](https://onlinelibrary.wiley.com/doi/full/10.1111/itor.13527)
1. [Improving the efficiency of last-mile delivery with the flexible drones traveling salesman problem](https://www.sciencedirect.com/science/article/pii/S0957417422014701)
1. [**Facility location decisions for drone delivery with riding: A literature review**](https://www.sciencedirect.com/science/article/pii/S0305054824001448)
1. [**Maximum coverage capacitated facility location problem with range constrained drones**](https://www.sciencedirect.com/science/article/pii/S0968090X18307575)
1. [**Robust Optimization for Supply Chain Applications: Facility Location and Drone Delivery Problems**](https://www.proquest.com/openview/dce38235e6cecdb0dd4b6493d6e487d0/1?cbl=18750&diss=y&pq-origsite=gscholar)
1. [**Robust Maximum Coverage Facility Location Problem with Drones Considering Uncertainties in Battery Availability and Consumption**](https://journals.sagepub.com/doi/abs/10.1177/0361198120968094)
1. [**Facility Location Problem Approach for Distributed Drones**](https://www.mdpi.com/2073-8994/11/1/118)
