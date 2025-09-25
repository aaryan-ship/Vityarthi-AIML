#!/usr/bin/env python3
"""
delivery_agent.py

Single-file implementation for the 'Autonomous Delivery Agent' project.
Contains:
 - GridWorld: environment model (static obstacles, terrain costs, dynamic obstacles schedule)
 - SearchAlgorithms: BFS, UCS, A* (supports time-expanded search when deterministic dynamic obstacles present)
 - LocalSearch: Simulated Annealing to repair/replan paths quickly
 - DeliveryAgent: executes path step-by-step and triggers replanning on dynamic obstacles
 - CLI and experiment runner

Map file format (maps/*.txt):
 - Rows of tokens separated by spaces.
 - Tokens:
     S   = Start (treated as cost 1)
     G   = Goal  (treated as cost 1)
     #   = Static obstacle (impassable)
     .   = terrain cost 1
     2,3,... = integer terrain cost >=1
Example:
S . . # G
. . 2 . .
# . . . .

Dynamic schedule format (maps/*.dyn.json):
 - JSON list of obstacle objects: [{"id": "v1", "positions": [[t,x,y], [t,x,y], ...]}, ...]
 - positions list must be sorted by time. If an obstacle has no entry for time t, it is assumed not present at t.
 - Times are integers starting at 0 (agent start time).

Usage examples:
 - Plan on a static map with A*:
     python delivery_agent.py --map maps/small_map.txt --algo astar
 - Run experiment (all planners on all maps):
     python delivery_agent.py --run-experiments
"""
import argparse, heapq, time, math, random, os, json, csv, sys
from collections import deque, defaultdict

# ---------- GridWorld ----------
class GridWorld:
    def __init__(self, grid=None, start=None, goal=None, dynamic_schedule=None):
        # grid: 2D list where each cell is int cost or None for static obstacle
        self.grid = grid or []
        self.rows = len(self.grid)
        self.cols = len(self.grid[0]) if self.rows>0 else 0
        self.start = start
        self.goal = goal
        # dynamic_schedule: list of obstacle dicts as loaded from JSON schedule file
        # We'll also build a time->set((r,c)) map for fast lookup
        self.dynamic_schedule = dynamic_schedule or []
        self.time_occupancy = defaultdict(set)  # time -> set of (r,c)
        self._build_time_occupancy()

    def _build_time_occupancy(self):
        self.time_occupancy = defaultdict(set)
        for obs in self.dynamic_schedule:
            obs_id = obs.get("id", "?")
            for entry in obs.get("positions", []):
                if len(entry) >= 3:
                    t, r, c = entry[0], entry[1], entry[2]
                    self.time_occupancy[int(t)].add((int(r), int(c)))

    @classmethod
    def from_map_file(cls, path_map, path_dyn=None):
        grid = []
        start=None
        goal=None
        with open(path_map, 'r') as f:
            for r,line in enumerate(f):
                line=line.strip()
                if not line: continue
                tokens=line.split()
                row=[]
                for c,tk in enumerate(tokens):
                    if tk.upper()=='S':
                        start=(r,c)
                        row.append(1)
                    elif tk.upper()=='G':
                        goal=(r,c)
                        row.append(1)
                    elif tk=='#':
                        row.append(None)
                    elif tk=='.':
                        row.append(1)
                    else:
                        try:
                            val=int(tk)
                            if val<1:
                                raise ValueError()
                            row.append(val)
                        except:
                            # unknown token -> treat as obstacle
                            row.append(None)
                grid.append(row)
        dyn=[]
        if path_dyn and os.path.exists(path_dyn):
            with open(path_dyn,'r') as f:
                dyn = json.load(f)
        gw = cls(grid, start, goal, dyn)
        return gw

    def in_bounds(self, r,c):
        return 0<=r<self.rows and 0<=c<self.cols

    def is_static_free(self, r,c):
        return self.in_bounds(r,c) and (self.grid[r][c] is not None)

    def is_free_at(self, r,c, t):
        # Checks static and dynamic occupancy at time t
        if not self.is_static_free(r,c):
            return False
        occ = self.time_occupancy.get(t, set())
        return (r,c) not in occ

    def neighbors(self, r,c, allow_diagonal=False):
        dirs = [(1,0),(-1,0),(0,1),(0,-1)]
        if allow_diagonal:
            dirs += [(1,1),(1,-1),(-1,1),(-1,-1)]
        for dr,dc in dirs:
            nr, nc = r+dr, c+dc
            if self.in_bounds(nr,nc) and self.grid[nr][nc] is not None:
                yield nr,nc

    def terrain_cost(self, r,c):
        # movement cost to ENTER cell (r,c)
        return self.grid[r][c]

    def min_terrain_cost(self):
        m=math.inf
        for r in range(self.rows):
            for c in range(self.cols):
                v = self.grid[r][c]
                if v is not None and v < m:
                    m = v
        return m if m!=math.inf else 1

    def pretty_print(self, path=None, show_dynamic_time=None):
        pathset = set(path) if path else set()
        for r in range(self.rows):
            row=''
            for c in range(self.cols):
                if (r,c)==self.start:
                    row += 'S '
                elif (r,c)==self.goal:
                    row += 'G '
                elif (r,c) in pathset:
                    row += '* '
                elif self.grid[r][c] is None:
                    row += '# '
                else:
                    row += str(self.grid[r][c])+' '
            print(row)

# ---------- SearchAlgorithms ----------
class SearchResult:
    def __init__(self, path, cost, nodes_expanded, time_seconds):
        self.path = path
        self.cost = cost
        self.nodes_expanded = nodes_expanded
        self.time_seconds = time_seconds

def reconstruct_parents(parents, goal_state):
    path=[]
    cur=goal_state
    while cur in parents:
        path.append(cur[:2])
        cur = parents[cur]
    path.reverse()
    return path

class SearchAlgorithms:
    @staticmethod
    def bfs(gridworld, consider_time=False, time_horizon=500):
        start = gridworld.start
        goal = gridworld.goal
        t0 = time.time()
        nodes_expanded=0
        if not start or not goal:
            return SearchResult(None, math.inf, 0, 0.0)
        if consider_time:
            # state is (r,c,t)
            start_state = (start[0], start[1], 0)
            q = deque([start_state])
            parents = {}
            visited = set([start_state])
            while q:
                cur = q.popleft()
                r,c,t = cur
                nodes_expanded += 1
                if (r,c)==goal:
                    path = reconstruct_parents(parents, cur)
                    cost = sum(gridworld.terrain_cost(rr,cc) for rr,cc in path)
                    return SearchResult(path, cost, nodes_expanded, time.time()-t0)
                for nr,nc in gridworld.neighbors(r,c):
                    nt = t+1
                    if nt>time_horizon: continue
                    if not gridworld.is_free_at(nr,nc,nt): continue
                    nxt = (nr,nc,nt)
                    if nxt in visited: continue
                    visited.add(nxt)
                    parents[nxt]=cur
                    q.append(nxt)
            return SearchResult(None, math.inf, nodes_expanded, time.time()-t0)
        else:
            q = deque([start])
            parents = {}
            visited = set([start])
            while q:
                cur = q.popleft()
                nodes_expanded += 1
                if cur==goal:
                    path = reconstruct_parents({k: (v[0],v[1],0) for k,v in parents.items()}, (cur[0],cur[1],0))
                    cost = sum(gridworld.terrain_cost(rr,cc) for rr,cc in path)
                    return SearchResult(path, cost, nodes_expanded, time.time()-t0)
                for nr,nc in gridworld.neighbors(cur[0],cur[1]):
                    if (nr,nc) in visited: continue
                    visited.add((nr,nc))
                    parents[(nr,nc,0)] = (cur[0],cur[1],0)
                    q.append((nr,nc))
            return SearchResult(None, math.inf, nodes_expanded, time.time()-t0)

    @staticmethod
    def uniform_cost(gridworld, consider_time=False, time_horizon=500):
        start = gridworld.start
        goal = gridworld.goal
        t0=time.time()
        nodes_expanded=0
        if not start or not goal:
            return SearchResult(None, math.inf, 0, 0.0)
        if consider_time:
            start_state=(start[0],start[1],0)
            pq=[]
            heapq.heappush(pq,(0,start_state))
            parents={}
            best_cost={start_state:0}
            while pq:
                cost_so_far, cur = heapq.heappop(pq)
                if cost_so_far>best_cost.get(cur, math.inf): continue
                nodes_expanded+=1
                r,c,t = cur
                if (r,c)==goal:
                    path = reconstruct_parents(parents, cur)
                    return SearchResult(path, cost_so_far, nodes_expanded, time.time()-t0)
                for nr,nc in gridworld.neighbors(r,c):
                    nt = t+1
                    if nt>time_horizon: continue
                    if not gridworld.is_free_at(nr,nc,nt): continue
                    step_cost = gridworld.terrain_cost(nr,nc)
                    nxt=(nr,nc,nt)
                    newcost = cost_so_far + step_cost
                    if newcost < best_cost.get(nxt, math.inf):
                        best_cost[nxt]=newcost
                        parents[nxt]=cur
                        heapq.heappush(pq,(newcost,nxt))
            return SearchResult(None, math.inf, nodes_expanded, time.time()-t0)
        else:
            start_state=(start[0],start[1])
            pq=[(0,start_state)]
            parents={}
            best_cost={start_state:0}
            while pq:
                cost_so_far, cur = heapq.heappop(pq)
                if cost_so_far>best_cost.get(cur, math.inf): continue
                nodes_expanded+=1
                if cur==goal:
                    # rebuild parents mapping shape
                    path=[]
                    node = cur
                    while node in parents:
                        path.append(node)
                        node = parents[node]
                    path.append(start)
                    path.reverse()
                    return SearchResult(path, cost_so_far, nodes_expanded, time.time()-t0)
                for nr,nc in gridworld.neighbors(cur[0],cur[1]):
                    step_cost = gridworld.terrain_cost(nr,nc)
                    nxt=(nr,nc)
                    newcost = cost_so_far + step_cost
                    if newcost < best_cost.get(nxt, math.inf):
                        best_cost[nxt]=newcost
                        parents[nxt]=cur
                        heapq.heappush(pq,(newcost,nxt))
            return SearchResult(None, math.inf, nodes_expanded, time.time()-t0)

    @staticmethod
    def astar(gridworld, consider_time=False, time_horizon=500, allow_diagonal=False):
        start = gridworld.start
        goal = gridworld.goal
        t0=time.time()
        nodes_expanded=0
        if not start or not goal:
            return SearchResult(None, math.inf, 0, 0.0)
        minc = gridworld.min_terrain_cost()
        def heuristic(a,b):
            # Manhattan for 4-connected; if diagonal allowed use Chebyshev (approx)
            if allow_diagonal:
                dx = abs(a[0]-b[0]); dy = abs(a[1]-b[1])
                return max(dx,dy)*minc
            else:
                return (abs(a[0]-b[0]) + abs(a[1]-b[1]))*minc

        if consider_time:
            start_state=(start[0],start[1],0)
            gscore={start_state:0}
            fscore={start_state:heuristic(start,goal)}
            pq=[(fscore[start_state], start_state)]
            parents={}
            while pq:
                f, cur = heapq.heappop(pq)
                if f>fscore.get(cur, math.inf): continue
                nodes_expanded+=1
                r,c,t = cur
                if (r,c)==goal:
                    path = reconstruct_parents(parents, cur)
                    return SearchResult(path, gscore[cur], nodes_expanded, time.time()-t0)
                for nr,nc in gridworld.neighbors(r,c, allow_diagonal):
                    nt = t+1
                    if nt>time_horizon: continue
                    if not gridworld.is_free_at(nr,nc,nt): continue
                    step_cost = gridworld.terrain_cost(nr,nc)
                    nxt=(nr,nc,nt)
                    tentative_g = gscore[cur] + step_cost
                    if tentative_g < gscore.get(nxt, math.inf):
                        parents[nxt]=cur
                        gscore[nxt]=tentative_g
                        fscore[nxt]=tentative_g + heuristic((nr,nc), goal)
                        heapq.heappush(pq,(fscore[nxt],nxt))
            return SearchResult(None, math.inf, nodes_expanded, time.time()-t0)
        else:
            start_state=(start[0],start[1])
            gscore={start_state:0}
            fscore={start_state:heuristic(start,goal)}
            pq=[(fscore[start_state], start_state)]
            parents={}
            while pq:
                f, cur = heapq.heappop(pq)
                if f>fscore.get(cur, math.inf): continue
                nodes_expanded+=1
                if cur==goal:
                    # reconstruct path
                    path=[cur]
                    node=cur
                    while node in parents:
                        node = parents[node]
                        path.append(node)
                    path.reverse()
                    return SearchResult(path, gscore[cur], nodes_expanded, time.time()-t0)
                for nr,nc in gridworld.neighbors(cur[0],cur[1], allow_diagonal):
                    step_cost = gridworld.terrain_cost(nr,nc)
                    nxt=(nr,nc)
                    tentative_g = gscore[cur] + step_cost
                    if tentative_g < gscore.get(nxt, math.inf):
                        parents[nxt]=cur
                        gscore[nxt]=tentative_g
                        fscore[nxt]=tentative_g + heuristic(nxt, goal)
                        heapq.heappush(pq,(fscore[nxt],nxt))
            return SearchResult(None, math.inf, nodes_expanded, time.time()-t0)

# ---------- LocalSearch (Simulated Annealing) ----------
class LocalSearch:
    @staticmethod
    def path_cost(gridworld, path):
        return sum(gridworld.terrain_cost(r,c) for (r,c) in path)

    @staticmethod
    def simulated_annealing_repair(gridworld, current_path, time_limit=1.0, max_iters=200, allow_diagonal=False):
        """
        Try to repair or improve an existing path quickly using simulated annealing.
        Neighbor generation: pick two indices i<j along the path, attempt to connect path[i] to path[j]
        with a fresh A* (local re-solve). If it finds solution, replace the subpath.
        """
        import time as _time
        tstart=_time.time()
        best_path = list(current_path)
        best_cost = LocalSearch.path_cost(gridworld, best_path)
        if len(best_path)<3:
            return best_path, best_cost, 0, 0.0  # nothing to repair
        iter_count=0
        nodes_expanded_total=0
        # temperature schedule
        T0 = 1.0
        Tmin = 0.001
        for it in range(max_iters):
            iter_count+=1
            if _time.time()-tstart > time_limit:
                break
            # temperature decay
            T = max(Tmin, T0 * (1 - (it/max(1,max_iters-1))))
            # choose random segment to replace
            i = random.randint(0, len(best_path)-3)
            j = random.randint(i+2, min(len(best_path)-1, i+8))  # limit length of segment to replace
            a = best_path[i]
            b = best_path[j]
            # run A* between a and b on static map (time-agnostic)
            gw_temp = gridworld
            sub_gw = gw_temp  # same gridworld
            # create temporary gridworld with start=a and goal=b for astar call
            old_start, old_goal = sub_gw.start, sub_gw.goal
            sub_gw.start, sub_gw.goal = a, b
            res = SearchAlgorithms.astar(sub_gw, consider_time=False, allow_diagonal=allow_diagonal)
            sub_gw.start, sub_gw.goal = old_start, old_goal
            nodes_expanded_total += res.nodes_expanded
            if res.path is None:
                continue
            # build new candidate path: prefix + new_subpath (without duplicating endpoints) + suffix
            new_path = best_path[:i] + res.path + best_path[j+1:]
            # ensure new_path is valid (adjacent steps)
            valid = True
            for k in range(len(new_path)-1):
                if abs(new_path[k][0]-new_path[k+1][0])+abs(new_path[k][1]-new_path[k+1][1])>1:
                    # allow diagonal only if enabled
                    if not allow_diagonal:
                        valid=False; break
            if not valid:
                continue
            new_cost = LocalSearch.path_cost(gridworld, new_path)
            # accept if better, or with probability exp(-(Delta)/T)
            delta = new_cost - best_cost
            if delta < 0 or random.random() < math.exp(-delta / (T+1e-9)):
                best_path = new_path
                best_cost = new_cost
        return best_path, best_cost, nodes_expanded_total, _time.time()-tstart

# ---------- DeliveryAgent ----------
class DeliveryAgent:
    def __init__(self, gridworld, planner='astar', allow_diagonal=False, time_horizon=500, local_search=True):
        self.gridworld = gridworld
        self.planner = planner
        self.allow_diagonal=allow_diagonal
        self.time_horizon=time_horizon
        self.local_search = local_search
        self.log = []

    def plan(self, consider_time=False):
        if self.planner=='bfs':
            return SearchAlgorithms.bfs(self.gridworld, consider_time=consider_time, time_horizon=self.time_horizon)
        elif self.planner=='ucs':
            return SearchAlgorithms.uniform_cost(self.gridworld, consider_time=consider_time, time_horizon=self.time_horizon)
        else:
            return SearchAlgorithms.astar(self.gridworld, consider_time=consider_time, time_horizon=self.time_horizon, allow_diagonal=self.allow_diagonal)

    def execute(self, max_steps=1000, capture_trace=False):
        """
        Execute current plan from start to goal, monitoring dynamic obstacles.
        If an obstacle appears on the next step, try local-search repair; if that fails, replan with A* (time-aware if schedule known).
        Returns a SearchResult and optionally a trace list of positions at each timestep (including start).
        """
        # initial planning: if dynamic schedule exists, use time-aware planning; else time-agnostic
        consider_time = (len(self.gridworld.time_occupancy)>0)
        res = self.plan(consider_time=consider_time)
        if res.path is None:
            self.log.append("No initial path found.")
            return res, ( [self.gridworld.start] if capture_trace else None )
        path = res.path
        cost = res.cost
        nodes_expanded_total = res.nodes_expanded
        t = 0
        executed_path=[self.gridworld.start]
        steps=0
        trace = [self.gridworld.start] if capture_trace else None
        # ensure path begins with start
        if path[0] != self.gridworld.start:
            path = [self.gridworld.start] + path
        while path and steps<max_steps:
            steps+=1
            if len(path)>1:
                next_cell = path[1]
            else:
                next_cell = path[0]
            arrival_t = t+1
            # check if next_cell is free at arrival_t
            if not self.gridworld.is_free_at(next_cell[0], next_cell[1], arrival_t):
                # dynamic obstacle blocks next move
                msg = f"[t={t}] Obstacle blocks next cell {next_cell} at t={arrival_t} -> replanning"
                self.log.append(msg)
                # try local search repair (time-agnostic repair to quickly avoid unknown-most obstacles)
                if self.local_search:
                    repaired_path, rep_cost, rep_nodes, rep_time = LocalSearch.simulated_annealing_repair(self.gridworld, path, time_limit=0.5, max_iters=100, allow_diagonal=self.allow_diagonal)
                    nodes_expanded_total += rep_nodes
                    if repaired_path and repaired_path[0]==path[0] and repaired_path[-1]==path[-1] and repaired_path!=path:
                        self.log.append(f"Local search repaired path in {rep_time:.3f}s (nodes_expanded={rep_nodes})")
                        path = repaired_path
                        cost = rep_cost
                        continue
                # fallback: replan using A* time-aware if schedule known else time-agnostic
                if consider_time:
                    old_start, old_goal = self.gridworld.start, self.gridworld.goal
                    self.gridworld.start = path[0]
                    res2 = SearchAlgorithms.astar(self.gridworld, consider_time=True, time_horizon=self.time_horizon, allow_diagonal=self.allow_diagonal)
                    self.gridworld.start, self.gridworld.goal = old_start, old_goal
                else:
                    res2 = SearchAlgorithms.astar(self.gridworld, consider_time=False, allow_diagonal=self.allow_diagonal)
                nodes_expanded_total += res2.nodes_expanded
                if res2.path:
                    path = res2.path
                    cost = res2.cost
                    self.log.append(f"Replanned with A* (nodes={res2.nodes_expanded}, cost={res2.cost:.2f})")
                    continue
                else:
                    self.log.append("Replanning failed. No path found.")
                    return SearchResult(None, math.inf, nodes_expanded_total, 0.0), trace
            # move to next_cell
            executed_path.append(next_cell)
            t += 1
            if capture_trace:
                trace.append(next_cell)
            # advance path one step
            path = path[1:]
            if len(path)==1 and path[0]==self.gridworld.goal:
                self.log.append(f"Reached goal at t={t}. Total nodes_expanded={nodes_expanded_total}")
                final_path = executed_path
                return SearchResult(final_path, LocalSearch.path_cost(self.gridworld, final_path), nodes_expanded_total, 0.0), trace
        self.log.append("Max steps exceeded or aborted.")
        return SearchResult(None, math.inf, nodes_expanded_total, 0.0), trace

# ---------- Utilities ----------
def write_sample_maps(base_dir='maps'):
    os.makedirs(base_dir, exist_ok=True)
    # small map 5x5
    small = """S . . . G
. # # . .
. . 2 . .
. # . . .
. . . . .
"""
    with open(os.path.join(base_dir,'small_map.txt'),'w') as f:
        f.write(small)
    dyn_small = [
        {"id":"v1", "positions":[ [1,0,2], [2,0,3] ] }
    ]
    with open(os.path.join(base_dir,'small_map.dyn.json'),'w') as f:
        json.dump(dyn_small,f,indent=2)

if __name__=='__main__':
    write_sample_maps('maps')
    print("Sample maps written to maps/")    
