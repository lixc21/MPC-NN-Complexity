import numpy
import hashlib


class LTISystem:
    """x_{t+1} = A x_t + B u_t, G x_t <= g, H u_t <= h"""
    def __init__(self, 
                 A: numpy.ndarray, B: numpy.ndarray, 
                 G: numpy.ndarray, g: numpy.ndarray, 
                 H: numpy.ndarray, h: numpy.ndarray, x0_limit: numpy.ndarray):
        self.A = A
        self.B = B
        self.G = G
        self.g = g
        self.H = H
        self.h = h
        self.x0_limit = x0_limit
        self.state_dim = A.shape[0]
        self.input_dim = B.shape[1]
    
    def get_random_x0(self):
        return numpy.array([numpy.random.uniform(-b, b) for b in self.x0_limit])
    
    def get_hash(self):
        if not hasattr(self, "hash"):
            hash_input = (self.A.tobytes(), self.B.tobytes(), 
                          self.G.tobytes(), self.g.tobytes(), 
                          self.H.tobytes(), self.h.tobytes(),
                          tuple(self.x0_limit))
            hash_md5 = hashlib.md5()
            hash_md5.update(str(hash_input).encode('utf-8'))
            self.hash = hash_md5.hexdigest()
        return self.hash


if __name__ == "__main__":
    sys = LTISystem(
        A=numpy.array([[1, 1], [0, 1]]),
        B=numpy.array([[0.5], [1]]),
        G=numpy.array([[1, 0], [-1, 0], [0, 1], [0, -1]]),
        g=numpy.array([[25], [25], [5], [5]]),
        H=numpy.array([[1], [-1]]),
        h=numpy.array([[1], [1]]),
        x0_limit=numpy.array([25, 5]))
    print(f"get_random_x0: {sys.get_random_x0()}")
    print(f"get_hash: {sys.get_hash()}")
