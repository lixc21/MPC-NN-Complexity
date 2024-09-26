import copy
import numpy
import gurobipy
import hashlib
import matplotlib.pyplot as pyplot
from LTISystem import LTISystem
import scipy.linalg


class LinearMPC:
    """
    min_u   sum_{t=0}^{N-1} x_t^T Q x_t + u_t^T R u_t + x_N^T P x_N  
    s.t.    x_{t+1} = A x_t + B u_t, G x_t <= g, H u_t <= h  
            final_G x_N <= final_g (optional)  
    """
    def __init__(self, sys: LTISystem, 
                 Q: numpy.ndarray, R: numpy.ndarray, N: int, P: numpy.ndarray | None = None, 
                 final_G: numpy.ndarray | None = None, final_g: numpy.ndarray | None = None):
        self.sys = sys
        self.Q = Q
        self.R = R
        self.N = N
        self.P = P
        self.P = Q if P is None else P
        self.final_G = final_G
        self.final_g = final_g
        self.solver = self.get_solver()


    @classmethod
    def from_mpc_config(cls, mpc_config):
        return cls(mpc_config["sys"], mpc_config["Q"], mpc_config["R"], mpc_config["N"], 
                   mpc_config["P"], mpc_config["final_G"], mpc_config["final_g"])

    def to_mpc_config(self):
        return {"sys": self.sys, "Q": self.Q, "R": self.R, "N": self.N, "P": self.P, 
                "final_G": self.final_G, "final_g": self.final_g}

    def copy(self):
        return LinearMPC.from_mpc_config(copy.deepcopy(self.to_mpc_config()))

    def use_dare_P(self):
        self.P = scipy.linalg.solve_discrete_are(self.sys.A, self.sys.B, self.Q, self.R)
        print(f"calculate P using DARE, P:")
        print(self.P)

    def simulate(self, x0=None, step_count=None):
        x0 = x0 if x0 is not None else self.get_random_feasible_x0()
        step_count = step_count if step_count is not None else self.N * 2
        print(f"simulating the MPC controller with initial state {x0} for {step_count} steps...")
        trajectory = []
        x = x0
        u = self.solve(x)
        trajectory.append((x, u))
        for _ in range(step_count):
            x = self.sys.A @ x + self.sys.B @ u
            u = self.solve(x)
            trajectory.append((x, u))
        for i, (x, u) in enumerate(trajectory):
            print(f"Step {i}: x={x}, u={u}")
        # plot the trajectory
        x_values = [x for x, _ in trajectory]
        u_values = [u for _, u in trajectory]
        fig, (ax1, ax2) = pyplot.subplots(2, 1, figsize=(10, 8))
        time = range(len(trajectory))
        for i in range(len(x_values[0])):
            ax1.plot(time, [x[i] for x in x_values], label=f'x{i+1}')
        for i in range(len(u_values[0])):
            ax1.plot(time, [u[i] for u in u_values], label=f'u{i+1}')
        ax1.set_xlabel('Time')
        ax1.legend()
        # Plotting norm of x and u vs time
        x_norm = [numpy.linalg.norm(x) for x, _ in trajectory]
        u_norm = [numpy.linalg.norm(u) for _, u in trajectory]
        ax2.plot(time, x_norm, label='||x||')
        ax2.plot(time, u_norm, label='||u||')
        ax2.set_xlabel('Time')
        ax2.set_yscale('log')
        ax2.legend()
        pyplot.tight_layout()
        pyplot.show()

    def get_hash(self):
        if not hasattr(self, "hash"):
            hash_input = (self.sys.get_hash(), self.Q.tobytes(), self.R.tobytes(), self.N, 
                          self.P.tobytes() if self.P is not None else None,
                          self.final_G.tobytes() if self.final_G is not None else None,
                          self.final_g.tobytes() if self.final_g is not None else None)
            hash_md5 = hashlib.md5()
            hash_md5.update(str(hash_input).encode('utf-8'))
            self.hash = hash_md5.hexdigest()
        return self.hash

    def get_solver(self):
        opt = gurobipy.Model("mpc")
        opt.setParam('OutputFlag', 0)
        opt.setParam('Threads', 1)
        # variables (default lb is 0)
        self.x = opt.addMVar(shape=(self.sys.state_dim, self.N+1), name="x", lb=-gurobipy.GRB.INFINITY)
        self.u = opt.addMVar(shape=(self.sys.input_dim, self.N), name="u", lb=-gurobipy.GRB.INFINITY)
        # objective function
        obj = gurobipy.QuadExpr()
        for t in range(self.N):
            obj += self.x[:, t] @ self.Q @ self.x[:, t] + self.u[:, t] @ self.R @ self.u[:, t]
        obj += self.x[:, self.N] @ self.P @ self.x[:, self.N]
        opt.setObjective(obj, gurobipy.GRB.MINIMIZE)
        # constraints
        for t in range(self.N):
            opt.addConstr(self.x[:, t+1] == self.sys.A @ self.x[:, t] + self.sys.B @ self.u[:, t], name=f"sys_{t}")
            opt.addConstr(self.sys.G @ self.x[:, t+1] <= self.sys.g, name=f"state_{t}")
            opt.addConstr(self.sys.H @ self.u[:, t] <= self.sys.h, name=f"input_{t}")
        if self.final_G is not None:
            opt.addConstr(self.final_G @ self.x[:, self.N] <= self.final_g, name="final_constraint")
        # initial condition (flatten to 1D array for Gurobi)
        x0 = self.sys.get_random_x0().flatten()
        self.init_cons = opt.addConstr(self.x[:, 0] == x0, name="init_condition")
        return opt

    def get_random_feasible_x0(self):
        while True:
            x0 = self.sys.get_random_x0()
            if self.solve(x0) is not None:
                return x0

    def solve(self, x0: numpy.ndarray):
        self.init_cons.rhs = x0.flatten()
        self.solver.optimize()
        if self.solver.status == gurobipy.GRB.OPTIMAL:
            return self.u.X[:, 0]
        return None


if __name__ == "__main__":
    print("Testing LinearMPC...")
    final_g = numpy.ones((4, 1))
    final_G = numpy.array([[0.356859320267763, 0.118391010392355],
                           [-0.356859320267763, -0.118391010392355],
                           [-0.616695261517283, -1.27031632620086],
                           [0.616695261517283, 1.27031632620086]])
    sys = LTISystem(
        A=numpy.array([[1, 1], [0, 1]]),
        B=numpy.array([[0.5], [1]]),
        G=numpy.array([[1, 0], [-1, 0], [0, 1], [0, -1]]),
        g=numpy.array([[25], [25], [5], [5]]),
        H=numpy.array([[1], [-1]]),
        h=numpy.array([[1], [1]]),
        x0_limit=numpy.array([25, 5]))
    mpc = LinearMPC(sys,
                    Q=numpy.eye(2),
                    R=numpy.eye(1) * 0.1,
                    P=numpy.array([[2.05987690431647, 0.591607978309962], 
                                   [0.591607978309962, 1.42283562177507]]),
                    N=10)
    print(f"get_hash: {mpc.get_hash()}")
    print(f"calculate u by Python: {mpc.solve(numpy.array([0.1, 0.1]))}")
    print(mpc.solve(numpy.array([0.1, 0.1])))
    mpc = LinearMPC(sys,
                    Q=numpy.eye(2),
                    R=numpy.eye(1) * 0.1,
                    P=numpy.array([[2.05987690431647, 0.591607978309962], 
                                   [0.591607978309962, 1.42283562177507]]),
                    final_G=final_G, final_g=final_g,
                    N=10)
    # should equal to -0.188701123867292 (MATLAB)
    print(f"get_hash: {mpc.get_hash()}")
    print(f"calculate u by MATLAB: {-0.188701123867292}")
    print(f"calculate u by Python: {mpc.solve(numpy.array([0.1, 0.1]))}")
    mpc.simulate()

    # test copying
    print(f"mpc get_hash: {mpc.get_hash()}")
    mpc2 = mpc.copy()
    print(f"mpc2 get_hash: {mpc2.get_hash()}")
    mpc3 = LinearMPC.from_mpc_config(mpc.to_mpc_config())
    print(f"mpc3 get_hash: {mpc3.get_hash()}")


