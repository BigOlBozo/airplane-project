import os
from time import perf_counter_ns as click
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
data = []
def run(learnrate, numiter):
    time = click()
    errors = 0
    diffs = []
    with open("Cancer_Data.txt", "r") as f:
        training_data = [parse_line(line) for line in f.readlines() if len(line) > 4]
    td = normalize(training_data)
    nn = NeuralNet(30,5,1)
    nn.train(td, iters=numiter, print_interval=numiter, learning_rate=learnrate)
    for i in nn.test_with_expected(td):
        if int(i[1][0]) == 1:
            if (1-(float(i[2][0]))) > 0.01:
                errors += 1
                #print(f'expected: {i[1][0]} actual:{float(i[2][0])}')
            diffs.append(1-float(i[2][0]))

        if int(i[1][0]) == 0:
            if float(i[2][0]) > 0.01:
                errors +=1
                #print(f'expected: {i[1][0]} actual:{float(i[2][0])}')
            diffs.append(float(i[2][0]))
    
    #data.append([numiter,learnrate,errors,time])    
    print(f'rate: {learnrate} errors: {errors} avg diff: {sum(diffs)/len(diffs)} time: {(int(click())-int(time))/(1000000000)}')
    elapsed = (int(click())-int(time))/(1000000000)
    data.append([numiter,learnrate,errors,elapsed])


for y in range(1,11):
    for x in range(1,11):
        run(float(x/10), y*100)
ratio = []      
for point in data:
    ratio.append((point[2])/float(point[3]))
for x in range(len(ratio)):
    if ratio[x] == min(ratio):
        print(data[x])
        break