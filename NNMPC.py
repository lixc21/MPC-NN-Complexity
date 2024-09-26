import logging
import os
import numpy
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as pyplot
from LTISystem import LTISystem
from LinearMPC import LinearMPC
from NN import FullyConnectedNN, ResNet
import multiprocessing
import time



class NNMPC:
    def __init__(self, mpc: LinearMPC, network: torch.nn.Module,
                 device: str | None = None, data_dir : str = './data'):
        self.mpc = mpc
        self.network = network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else torch.device(device)
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        # avoid multi-threading in PyTorch
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)


    @staticmethod
    def gen_dataset_worker(mpc_config: dict, num_sample: int, data_dir: str, process_id: int):
        numpy.random.seed(process_id * 100 + int(time.time()))
        worker_mpc = LinearMPC.from_mpc_config(mpc_config)
        data_x = numpy.zeros((num_sample, worker_mpc.sys.state_dim))
        data_u = numpy.zeros((num_sample, worker_mpc.sys.input_dim))
        index = num_sample
        while index > 0:
            x0 = worker_mpc.sys.get_random_x0()
            u0 = worker_mpc.solve(x0)
            if u0 is not None:
                data_x[-index] = x0
                data_u[-index] = u0
                index -= 1
        numpy.savez(f'{data_dir}/dataset_mpc_{worker_mpc.get_hash()}.{process_id}.npz', x=data_x, u=data_u)

    def gen_dataset(self, num_sample: int, num_worker: int = 1):
        start_time = time.time()
        # generate dataset using multiple processes
        process_list = []
        for index in range(num_worker):
            num_sample_this_worker = num_sample // num_worker
            remaining_sample = num_sample % num_worker
            if index < remaining_sample:
                num_sample_this_worker += 1
            process = multiprocessing.Process(target=self.gen_dataset_worker, 
                      args=(self.mpc.to_mpc_config(), num_sample_this_worker, self.data_dir, index))
            process.start()
            process_list.append(process)
        for process in process_list:
            process.join()
        logging.info(f"generate {num_sample} samples using {num_worker} workers, time: {time.time() - start_time}s")
        # merge dataset
        data_x = numpy.zeros((num_sample, self.mpc.sys.state_dim))
        data_u = numpy.zeros((num_sample, self.mpc.sys.input_dim))
        num_sample_loaded = 0
        for index in range(num_worker):
            with numpy.load(f'{self.data_dir}/dataset_mpc_{self.mpc.get_hash()}.{index}.npz') as data:
                num_sample_this_worker = data['x'].shape[0]
                data_x[num_sample_loaded : num_sample_loaded + num_sample_this_worker] = data['x']
                data_u[num_sample_loaded : num_sample_loaded + num_sample_this_worker] = data['u']
                num_sample_loaded += num_sample_this_worker
        numpy.savez(f'{self.data_dir}/dataset_mpc_{self.mpc.get_hash()}.npz', x=data_x, u=data_u)
        for index in range(num_worker):
            os.remove(f'{self.data_dir}/dataset_mpc_{self.mpc.get_hash()}.{index}.npz')
        logging.info(f"saved to {self.data_dir}/dataset_mpc_{self.mpc.get_hash()}.npz")
    
    def get_dataset(self, num_samples: int):
        if not hasattr(self, "dataset"):
            assert os.path.exists(f'./data/dataset_mpc_{self.mpc.get_hash()}.npz'), "dataset not found, please generate it first."
            data = numpy.load(f'./data/dataset_mpc_{self.mpc.get_hash()}.npz')
            self.dataset = (data['x'], data['u'])
        data_x, data_u = self.dataset
        if data_x.shape[0] < num_samples:
            logging.error(f"dataset has {data_x.shape[0]} samples, less than required {num_samples}.")
            raise ValueError("dataset has less samples than required.")
        elif data_x.shape[0] > num_samples:
            data_x = data_x[:num_samples]
            data_u = data_u[:num_samples]
        return data_x, data_u

    def solve(self, x0: numpy.ndarray):
        x0 = torch.tensor(x0, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            u0 = self.network(x0)
        return u0.cpu().numpy()

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def train(self, 
                batch_size: int,
                num_samples: int,
                scheduler_gamma: float,
                learning_rate: float,
                TEST_DATASET_SIZE: int = 5000):
        # get dataset
        dataset = self.get_dataset(num_samples + TEST_DATASET_SIZE)
        data_x, data_u = dataset
        data_x = torch.tensor(data_x, dtype=torch.float32)
        data_u = torch.tensor(data_u, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(data_x, data_u)
        train_dataset = torch.utils.data.Subset(dataset, range(TEST_DATASET_SIZE, TEST_DATASET_SIZE + num_samples))
        test_dataset = torch.utils.data.Subset(dataset, range(TEST_DATASET_SIZE))
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        # nn initialization
        self.network.apply(self.init_weights)
        self.network.to(self.device)
        # train and test
        loss = nn.MSELoss()
        optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=scheduler_gamma)
        test_loss_history = []
        train_loss_history = []
        epoch = 0
        for epoch in range(5):
            # train
            self.network.train()
            train_loss = 0
            for x, u in self.train_loader:
                x = x.to(self.device)
                u = u.to(self.device)
                optimizer.zero_grad()
                u_pred = self.network(x)
                loss_val = loss(u_pred, u)
                loss_val.backward()
                optimizer.step()
                train_loss += loss_val.item()
            train_loss_history.append(train_loss / len(self.train_loader))
            # test
            self.network.eval()
            test_loss = 0
            with torch.no_grad():
                for x, u in self.test_loader:
                    x = x.to(self.device)
                    u = u.to(self.device)
                    u_pred = self.network(x)
                    test_loss += loss(u_pred, u).item()
            # update scheduler
            scheduler.step()
            test_loss_history.append(test_loss / len(self.test_loader))
            # early stop if the loss does not decrease for 20 epochs
            # if len(loss_history) >= 2:
            #     if loss_history[-1] >= min(loss_history[:-1]) * 0.95:
            #         if len(loss_history) >= 4:
            #             if all([loss_history[-i] > min(loss_history) * 0.95 for i in range(1, 5)]):
            #                 print(loss_history)
            #                 break
            # if epoch > 0: break # for testing
        logging.info(f"Total epoch: {epoch}, best train loss: {min(train_loss_history)}, best test loss: {min(test_loss_history)}")
        return min(train_loss_history), min(test_loss_history)


    def test_simulation(self, step_count: int | None = None):
        if step_count is None:
            step_count = self.mpc.N * 2
        data_x, data_u = self.get_dataset(step_count)
        fail_count = 0
        cost_sum = 0
        for x0 in data_x:
            u0 = self.mpc.solve(x0)
            if u0 is not None:
                trajectory = [(x0,u0)]
                cost_sum += x0.T @ self.mpc.Q @ x0 + u0.T @ self.mpc.R @ u0
                for t in range(step_count):
                    x_next = self.mpc.sys.A @ trajectory[-1][0] + self.mpc.sys.B @ trajectory[-1][1]
                    u_next = self.solve(x_next)
                    trajectory.append((x_next, u_next))
                    cost_sum += x_next.T @ self.mpc.Q @ x_next + u_next.T @ self.mpc.R @ u_next
                    if self.mpc.solve(x_next) is None:
                        fail_count += 1
                        break
            else:
                fail_count += 1
        fail_rate = fail_count / len(data_x)
        if len(data_x) == fail_count:
            avg_cost = numpy.inf
        else:
            avg_cost = cost_sum / (step_count * (len(data_x) - fail_count))
        return fail_rate, avg_cost
            




if __name__ == "__main__":
    if not os.path.exists('./train_log'):
        os.mkdir('./train_log')
    # initialize logging
    log_file = './train_log/{}.log'.format(time.strftime("%Y%m%d"))
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s - %(levelname)s] %(message)s')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('[%(asctime)s - %(levelname)s] %(message)s'))
    logging.getLogger('').addHandler(file_handler)

    # problem configuration
    sys = LTISystem(
        A=numpy.array([[1, 1], [0, 1]]),
        B=numpy.array([[0.5], [1]]),
        G=numpy.array([[1, 0], [-1, 0], [0, 1], [0, -1]]),
        g=numpy.array([[25], [25], [5], [5]]),
        H=numpy.array([[1], [-1]]),
        h=numpy.array([[1], [1]]),
        x0_limit=numpy.array([25, 5]))
    final_g = numpy.ones((4, 1))
    final_G = numpy.array([[0.356859320267763, 0.118391010392355],
                           [-0.356859320267763, -0.118391010392355],
                           [-0.616695261517283, -1.27031632620086],
                           [0.616695261517283, 1.27031632620086]])
    mpc = LinearMPC(sys,
                    Q=numpy.eye(2),
                    R=numpy.eye(1) * 0.1,
                    P=numpy.array([[2.05987690431647, 0.591607978309962], 
                                   [0.591607978309962, 1.42283562177507]]),
                    final_G=final_G, final_g=final_g, N=10)
    
    # nnmpc
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    network = FullyConnectedNN(input_dim=sys.state_dim, output_dim=sys.input_dim,
                                hidden_width=16, hidden_depth=2)
    nnmpc = NNMPC(mpc, network, device)

    # generate dataset test
    start_time = time.time()
    nnmpc.gen_dataset(num_sample=510000, num_worker=128)
    logging.info(f"generate dataset using 128 workers, time: {time.time() - start_time} ")
    nnmpc.get_dataset(510000); logging.info(f"load 20000 samples")
    try:
        nnmpc.get_dataset(510001)
    except Exception as e:
        logging.info(f"catch exception: {e}")

    # train test
    start_time = time.time()
    best_loss = nnmpc.train(batch_size=32, num_samples=10000, scheduler_gamma=0.9, learning_rate=0.01)
    logging.info(f"Time to train: {time.time() - start_time}, best_loss: {best_loss}"); start_time = time.time()
    fail_rate, avg_cost = nnmpc.test_simulation()
    logging.info(f"Time to test: {time.time() - start_time}, fail_rate: {fail_rate}, avg_cost: {avg_cost}"); start_time = time.time()

    
