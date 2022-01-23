import numpy as np

# b = np.load('resources/arrays.npz')
# print(b)

data = np.load('resources/arrays.npz', allow_pickle=True)
results = np.load('resources/myresult.npz', allow_pickle=True)
lst = data.files
a = data['a'] * 10000
f = data['f']
r = results['f']
print(lst)
hat = 1e-5
for item in lst:
    print(item)
    print(data[item])
print(hat)
print('yo')