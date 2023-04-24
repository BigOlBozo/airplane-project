import os
os.system('cls')
data = []
with open("airplaneminus.csv") as f:
    lines = 0
    for line in f:
        data.append(line.split(','))

sel = data[99]
print(sel)
print(len(data))
for num in sel:
    try:
        print(float(num))
    except:
        data.remove(sel)
print(len(data))
print(data)

