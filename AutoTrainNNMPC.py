import queue
import numpy
import time
import itertools
from LTISystem import LTISystem
from LinearMPC import LinearMPC
import threading
import multiprocessing
import time
import random
from LinearMPC import LinearMPC


class AutoTrainNNMPC:
    def __init__(self, 
            mpc: LinearMPC,
            device: str,
            nn_config: dict):
        self.device = device
        self.mpc = mpc.copy()
        self.nn_config = nn_config

    @staticmethod
    def batch_train_worker( 
                mpc_config: dict, 
                nn_config: dict,  
                lock: threading.Lock, 
                device: str,
                thread_id: int, 
                input_queue: multiprocessing.Queue, 
                output_queue: multiprocessing.Queue) -> None:
        from NN import FullyConnectedNN, ResNet
        from NNMPC import NNMPC
        from LinearMPC import LinearMPC
        import time
        # initialize logging
        import logging
        log_format = '[%(asctime)s - %(levelname)s] %(message)s'
        log_file = './train_log/{}.log'.format(time.strftime("%Y%m%d"))
        logging.basicConfig(level=logging.INFO, format=log_format)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger('').addHandler(file_handler)
        # define the nnmpc
        mpc = LinearMPC.from_mpc_config(mpc_config)
        sys = mpc.sys
        match nn_config["type"]:
            case "fully_connected":
                network = FullyConnectedNN(
                    input_dim=sys.state_dim, output_dim=sys.input_dim,
                    hidden_width=nn_config["hidden_width"],
                    hidden_depth=nn_config["hidden_depth"])
            case "resnet":
                network = ResNet(
                    input_dim=sys.state_dim, output_dim=sys.input_dim,
                    hidden_width=nn_config["hidden_width"],
                    num_blocks=nn_config["num_blocks"])
            case _:
                raise ValueError("Invalid neural network type")
        nnmpc = NNMPC(mpc, network, device)
        # train the nnmpc with different hyperparameters
        idle_count = 0
        while True:
            if idle_count > 2:
                break
            # get training hyperparameters
            try: 
                with lock: 
                    train_config = input_queue.get_nowait()
                    logging.info(f"process: {thread_id}, get train_config: {train_config}, input_queue size: {input_queue.qsize()}")
            except Exception as e:
                train_config = None
            if train_config is None:
                time.sleep(0.3)
                idle_count += 1
                continue
            # train the nnmpc
            batch_size, num_samples, scheduler_gamma, learning_rate = train_config
            train_loss, test_loss = nnmpc.train(batch_size, num_samples, scheduler_gamma, learning_rate)
            result = {"nn_config": nn_config, "train_config": train_config, "train_loss": train_loss, "test_loss": test_loss}
            with lock: 
                with open(f'./data/result_{sys.get_hash()}.csv', 'a') as f:
                    f.write(f"{nn_config['hidden_width']},{nn_config['hidden_depth']},{train_loss},{test_loss},{train_config}\n")
                while True:
                    try:
                        output_queue.put(result, block=False)
                        logging.info(f"process: {thread_id}, put result: {result}, output_queue size: {output_queue.qsize()}")
                        break
                    except queue.Full:
                        logging.info(f"process: {thread_id}, output_queue is full, sleep for 0.1s")
                        time.sleep(0.1)
            logging.info(f"process: {thread_id}, train_config: {train_config}, train_loss: {train_loss}, test_loss: {test_loss}")
        logging.info(f"process: {thread_id}, idle_count: {idle_count}, exit")



    def parallel_train_core(self, train_config_list: list, parallel_num: int) -> list:
        random.shuffle(train_config_list)
        processes = []
        manager = multiprocessing.Manager()
        input_queue = manager.Queue()
        output_queue = manager.Queue()
        lock = multiprocessing.Lock()
        # put training hyperparameters into the queue
        for config in train_config_list:
            input_queue.put(config)
        for index in range(parallel_num):
            process = multiprocessing.Process(target=self.batch_train_worker,
                args=(self.mpc.to_mpc_config(), self.nn_config, lock, self.device, index, input_queue, output_queue))
            process.start()
            print(f"starting process {process}")
            processes.append(process)
        print(f"all processes started, {processes}")
        while True:
            all_finished = False
            for process in processes:
                if process.is_alive():
                    all_finished = False
                    print(f"process {process} is alive")
                    break
                else:
                    all_finished = True
            if all_finished:
                print("all processes finished")
                break
            print("sleeping for 1 second")
            time.sleep(1)
        result_list = []
        while output_queue.qsize() > 0:
            result = output_queue.get()
            result_list.append(result)
        # sort the result list
        result_list = sorted(result_list, key=lambda x: x["test_loss"])
        return result_list


    def parallel_train(self, parallel_num: int) -> dict:
        start_time = time.time()
        # hyperparameters zipping
        batch_size_list = [16, 32, 64]
        num_samples_list = [5000, 50000, 500000]
        # num_samples_list = [500] # debug
        scheduler_gamma_list = [0.1, 0.5, 0.9]
        learning_rate_list = [0.001, 0.01, 0.1]
        train_config_list = list(itertools.product(batch_size_list, num_samples_list, scheduler_gamma_list, learning_rate_list))
        train_config_list = train_config_list * 20
        random.shuffle(train_config_list)
        # train_config_list = train_config_list[:64] # debug
        print(f"Train config list length: {len(train_config_list)}")
        result_list = self.parallel_train_core(train_config_list, parallel_num)
        print(f"Best result: {result_list[0]}")
        print(f"Time elapsed: {time.time() - start_time}")
        return result_list[0]


if __name__ == "__main__":
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
    device = "cuda"
    device = "cpu"
    nn_config = {"type": "fully_connected", "hidden_width": 6, "hidden_depth": 2}



    # grid search
    with open(f'./data/best_result_{sys.get_hash()}.csv', 'a') as f:
        f.write("hidden_width,hidden_depth,loss,train_config\n")
    for hidden_width in [4, 6, 8, 10, 12]:
        for hidden_depth in [1, 2, 3, 4, 5]:
            nn_config["hidden_width"] = hidden_width
            nn_config["hidden_depth"] = hidden_depth
            auto_train = AutoTrainNNMPC(mpc, device, nn_config)
            best_result = auto_train.parallel_train(128)




