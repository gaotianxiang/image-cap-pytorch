import json
import logging
from tqdm import tqdm


class RunningAverage:
    def __init__(self):
        self.count = 0
        self.total = 0

    def update(self, num):
        self.count += 1
        self.total += num

    def reset(self):
        self.count = 0
        self.total = 0

    def __call__(self, *args, **kwargs):
        try:
            avg = self.total / self.count
        except ZeroDivisionError:
            avg = 'NaN'
        return avg


class Params:
    """Class that loads hyperparameters from a json file.
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


def running_average_test():
    ra = RunningAverage()
    print(ra())


def set_logger(log_path, terminal=True):
    """Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
        terminal: whether add console handler
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        if terminal:
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(logging.Formatter('%(message)s'))
            logger.addHandler(stream_handler)


def log(msg):
    logging.info(msg)
    tqdm.write(msg)


if __name__ == '__main__':
    running_average_test()
