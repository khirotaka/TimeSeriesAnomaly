import numpy as np


class Dataset:
    def __init__(self):
        t = np.linspace(0, 5*np.pi, 500)
        self.train = 10 * np.sin(t).reshape(-1, 1)
        self.train = np.tile(np.abs(self.train), (32, 1)).astype("f")

        t = np.linspace(0, 4*np.pi, 400)
        self.valid = 10 * np.sin(t).reshape(-1, 1)
        self.valid = np.concatenate((
            np.random.randn(100).reshape(100, 1), self.valid
        ), axis=0)

        self.valid = np.tile(np.abs(self.valid), (4, 1)).astype("f")

    @staticmethod
    def _divide_into_batches(data: np.ndarray, batch_size: int):
        n_time, n_dim_obs = data.shape
        nbatch = n_time // batch_size
        data = data[:nbatch * batch_size]
        data = data.reshape(batch_size, -1, n_dim_obs).transpose((1, 0, 2))
        return data

    @staticmethod
    def _get_batch(data, i, seq_len):
        slen = min(seq_len, data.shape[0] - i)
        inputs = data[i: i+slen]
        target = inputs.copy()
        return inputs, target
