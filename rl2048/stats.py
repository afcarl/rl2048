import pickle
import sys


class Stats:

    def __init__(self, modes=['train', 'val']):

        self.modes = {}
        for mode in modes:
            self.modes[mode] = {}

    def record(self, mode, name, value, index):

        stats = self.modes[mode]
        values = stats.get(name, [])
        values.append((index, value))
        stats[name] = values

    def plot(self):

        from matplotlib import pyplot as plt

        all_keys = []
        for mode in self.modes:
            all_keys.extend(self.modes[mode].keys())

        for key in set(all_keys):

            plt.figure()
            for mode in self.modes:
                if key in self.modes[mode]:
                    values = self.modes[mode][key]
                    values.sort()
                    xs = [val[0] for val in values]
                    ys = [val[1] for val in values]

                    plt.plot(xs, ys, label=mode)
                    plt.ylabel(key)
                    plt.grid(True)
                    plt.legend()

        plt.show()


if __name__ == '__main__':
    with open(sys.argv[1], 'rb') as f:
        stats = pickle.load(f)

        stats.plot()
