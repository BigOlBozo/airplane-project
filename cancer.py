import os
from typing import Tuple
from neural import *
os.system('cls')
file = []
def cleandata():  
    with open('Cancer_Data.csv','r+') as f:
        for line in f:
            fields = line.split(',')
            if len(fields) != '32':
                print(len(fields))
            id = fields[0]
            if fields[1] == 'M':
                fields[1] = '1'
            if fields[1] == 'B':
                fields[1] = '0'
            fields.remove(fields[0])
            string = (',').join(fields)
            file.append(string)
    with open('Cancer_Data.txt','w') as f:
        for line in file:
            f.write(line)

def parse_line(line: str) -> Tuple[List[float], List[float]]:
    """Splits line of CSV into inputs and output (transormfing output as appropriate)

    Args:
        line - one line of the CSV as a string

    Returns:
        tuple of input list and output list
    """
    tokens = line.split(",")
    out = int(tokens[0])
    output = [1 if out == 1 else 0]

    inpt = [float(x) for x in tokens[1:]]
    return (inpt, output)

def normalize(data: List[Tuple[List[float], List[float]]]):
    """Makes the data range for each input feature from 0 to 1

    Args:
        data - list of (input, output) tuples

    Returns:
        normalized data where input features are mapped to 0-1 range (output already
        mapped in parse_line)
    """
    leasts = len(data[0][0]) * [100.0]
    mosts = len(data[0][0]) * [0.0]

    for i in range(len(data)):
        for j in range(len(data[i][0])):
            if data[i][0][j] < leasts[j]:
                leasts[j] = data[i][0][j]
            if data[i][0][j] > mosts[j]:
                mosts[j] = data[i][0][j]

    for i in range(len(data)):
        for j in range(len(data[i][0])):
            data[i][0][j] = (data[i][0][j] - leasts[j]) / (mosts[j] - leasts[j])
    return data

def run():
    print('1')
    with open("Cancer_Data.txt", "r") as f:
        training_data = [parse_line(line) for line in f.readlines() if len(line) > 4]
    print('2')
    td = normalize(training_data)
    print('3')
    nn = NeuralNet(30,1,1)
    print('4')
    nn.train(td, iters=1_000, print_interval=10, learning_rate=0.5)
    print('5')
    for i in nn.test_with_expected(td):
        print(f"desired: {i[1]}, actual: {i[2]}")
run()