
import csv
import pandas as pd
import numpy as np

filename = "c:/whatever/file.csv"

with open(filename) as csvfile:
    reader = csv.reader(csvfile)

    for i, line in enumerate(reader):
        print(line)

        if i > 10:
            break


dtype_dict = {
    'col1': str,
    'col2': np.float32,
    'col3': np.int8
}

data = pd.read_csv(filename, dtype=dtype_dict)
