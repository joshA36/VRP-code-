# VRP-code-
import math
import requests
from collections import defaultdict
import gurobipy as gp
from gurobipy import GRB

URL = "https://vrp.atd-lab.inf.puc-rio.br/media/com_vrp/instances/A/A-n34-k5.vrp"  

# Solver parameters you can tune
TIME_LIMIT = 600            # seconds
MIP_GAP = 1e-6              # relative gap
VERBOSE = True              # show solver output

def read_vrp_from_url(url):
    r = requests.get(url)
    r.raise_for_status()
    lines = r.text.splitlines()
    coords, demands = {}, {}
    capacity, depot = None, None
    mode = None
    for raw in lines:
        line = raw.strip()
        if line == "" : continue
        up = line.upper()
        if up.startswith("CAPACITY"):
            capacity = int(line.split(":")[1])
        elif up.startswith("NODE_COORD_SECTION"):
            mode = "NODE"
            continue
        elif up.startswith("DEMAND_SECTION"):
            mode = "DEMAND"
            continue
        elif up.startswith("DEPOT_SECTION"):
            mode = "DEPOT"
            continue
        elif up.startswith("EOF"):
            break
        elif mode == "NODE":
            parts = line.split()
            if len(parts) >= 3:
                idx = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                coords[idx] = (x, y)
        elif mode == "DEMAND":
            parts = line.split()
            if len(parts) >= 2:
                idx = int(parts[0])
                d = int(parts[1])
                demands[idx] = d
        elif mode == "DEPOT":
            try:
                val = int(line.split()[0])
                if val != -1:
                    depot = val
            except:
                pass
    return coords, demands, depot, capacity

def euclidean(a, b):
    return int(round(math.hypot(a[0]-b[0], a[1]-b[1])))

def build_distance_matrix(coords):
    n = max(coords.keys())
    D = {}
    for i in range(1, n+1):
        for j in range(1, n+1):
            if i == j:
                D[i,j] = 0
            else:
                D[i,j] = euclidean(coords[i], coords[j])
    return D

def solve_cvrp_gurobi(coords, demands, depot, capacity, K, time_limit=TIME_LIMIT, mip_gap=MIP_GAP, verbose=VERBOSE):
    nodes = sorted(coords.keys())        # 1..n with depot included
    n = len(nodes)
    total_demand = sum(demands[i] for i in nodes if i != depot)

    # Distance matrix
    D = build_distance_matrix(coords)

    # Create model
    model = gp.Model("CVRP_flow")
    model.Params.TimeLimit = time_limit
    model.Params.MIPGap = mip_gap
    if not verbose:
        model.Params.OutputFlag = 0

# Binary arc variables x[i,j] for all i != j
    x = model.addVars(nodes, nodes, vtype=GRB.BINARY, name="x")
    # Flow variables f[i,j] >= 0 representing quantity of goods transported on arc (i->j)
    f = model.addVars(nodes, nodes, lb=0.0, name="f")

    # Objective: minimize total travel distance
    model.setObjective(gp.quicksum(D[i,j] * x[i,j] for i in nodes for j in nodes if i != j), GRB.MINIMIZE)

    # Degree constraints: each customer exactly one incoming and one outgoing
    for i in nodes:
        if i == depot:
            # depot: out-degree = K, in-degree = K
            model.addConstr(gp.quicksum(x[depot,j] for j in nodes if j != depot) == K, name=f"depot_out_deg")
            model.addConstr(gp.quicksum(x[j,depot] for j in nodes if j != depot) == K, name=f"depot_in_deg")
        else:
            model.addConstr(gp.quicksum(x[i,j] for j in nodes if j != i) == 1, name=f"out_deg_{i}")
            model.addConstr(gp.quicksum(x[j,i] for j in nodes if j != i) == 1, name=f"in_deg_{i}")

    # No self-arcs
    for i in nodes:
        model.addConstr(x[i,i] == 0)

    # Flow conservation (single-commodity) to eliminate subtours:
    # For customers i != depot: sum_j f[j,i] - sum_j f[i,j] == demands[i]
    for i in nodes:
        if i == depot:
            # Depot: net outgoing - incoming = -total_demand
            model.addConstr(gp.quicksum(f[depot,j] for j in nodes if j != depot) - gp.quicksum(f[j,depot] for j in nodes if j != depot) == total_demand, name="flow_depot")
        else:
            model.addConstr(gp.quicksum(f[j,i] for j in nodes if j != i) - gp.quicksum(f[i,j] for j in nodes if j != i) == demands[i], name=f"flow_cons_{i}")

    # Capacity linking: f[i,j] <= capacity * x[i,j] for all i!=j
    for i in nodes:
        for j in nodes:
            if i == j: 
                model.addConstr(f[i,j] == 0)
            else:
                model.addConstr(f[i,j] <= capacity * x[i,j], name=f"cap_link_{i}_{j}")

    # Additionally, flows from depot cannot exceed capacity per vehicle but total outgoing flow equals total demand handled across K vehicles
    # Implicitly ensured by cap_link and depot out-degree K.

    # (Optional) Symmetry breaking: prefer arcs out of depot with some ordering (not necessary but can help)
    # model.addConstr(x[depot, nodes[0]] >= x[depot, nodes[1]])

    # Optimize
    model.optimize()

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT or model.status == GRB.SUBOPTIMAL:
        obj = model.objVal
        print(f"\nObjective (total distance): {obj:.0f}")
        print(f"Model status: {model.status} (0=optimal, 9=time limit, ...)")
        # Extract arcs used
        used_arcs = [(i,j) for i in nodes for j in nodes if i != j and x[i,j].X > 0.5]
        print(f"Number of used arcs: {len(used_arcs)}")

        # Build adjacency map of used arcs
        succ = defaultdict(list)
        pred = defaultdict(list)
        for i,j in used_arcs:
            succ[i].append(j)
            pred[j].append(i)

        # Reconstruct routes: start from depot and follow arcs until back to depot, repeat until all arcs consumed
        routes = []
        used = set(used_arcs)
        for _ in range(K):
            if not any(a for a in used if a[0] == depot):  # no more depot-out arcs
                break
            # pick one available depot->j
            for arc in list(used):
                if arc[0] == depot:
                    current = arc[1]
                    used.remove(arc)
                    route = [depot, current]
                    break
            # follow until return to depot
            while route[-1] != depot:
                cur = route[-1]
                # find a successor arc (cur -> next) in used
                found_next = False
                for arc in list(used):
                    if arc[0] == cur:
                        next_node = arc[1]
                        used.remove(arc)
                        route.append(next_node)
                        found_next = True
                        break
                if not found_next:
                    # no outgoing arc from cur found (shouldn't happen), break
                    break
            routes.append(route)

        # Compute route loads and distances
        route_infos = []
        for r in routes:
            # r is list of nodes starting and ending at depot (hopefully)
            load = 0
            dist = 0
            for idx in range(len(r)-1):
                a, b = r[idx], r[idx+1]
                if b != depot:  # b could be depot at end; demands counted for customers only
                    load += demands.get(b, 0)
                dist += D[a,b]
            route_infos.append((r, load, dist))

        # Print routes
        for k, (r, load, dist) in enumerate(route_infos, 1):
            print(f"Vehicle {k}: route {r} | load {load} | distance {dist}")
        print(f"Total routes: {len(route_infos)}")
        return model, route_infos
    else:
        print("No feasible solution found or solver failed.")
        return model, None

def main():
    print("Reading instance from:", URL)
    coords, demands, depot, capacity = read_vrp_from_url(URL)
    print(f"Parsed: nodes={len(coords)}, depot={depot}, capacity={capacity}")
    # For A-n34-k5 the instance uses 5 vehicles (k=5). If different, adjust K accordingly.
    K = 5

    model, routes = solve_cvrp_gurobi(coords, demands, depot, capacity, K)
    if routes:
        total_dist = sum(r[2] for r in routes)
        print(f"\nSummed total distance (routes): {total_dist}")
    else:
        print("No routes to display.")

if __name__ == "__main__":
    main()

