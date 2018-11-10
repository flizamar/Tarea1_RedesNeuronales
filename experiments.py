import random
import itertools
from random import shuffle
from Red import *
import matplotlib.pyplot as plt
#Los presos definen la siguiente estrategia: si el primero dice 0, entonces significa que hay una cantidad par de ceros, si dice 1 significa que hay una cantidad impar
def paridad(arr):
  paridad = 0
  for i in arr:
    if i == 0:
      paridad = 1 - paridad
  return [paridad]

def resp_red(red, arr):
  return [1] if red.feed(arr) > 0.5 else [0]

def test_acc(red, test_set):
  cont = 0
  for i in range(len(test_set)):
    cont += 1 if resp_red(red, test_set[i]) == paridad(test_set[i]) else 0
  return cont

def main():
    n = 10
    epochs = 2000
    red = Red(n, [15, 10], 1)
    lst = list(map(list, itertools.product([0, 1], repeat=n)))
    test_size = 500
    shuffle(lst)
    test_set = lst[:test_size]
    train_set = lst[test_size:]
    results = []
    for j in range(epochs):
        for i in range(len(train_set)):
            red.train(paridad(train_set[i]), train_set[i], 0.5)
        print("Acc for epoch " + str(j) + ": " + str((test_acc(red, test_set)/test_size)*100))
        results.append((test_acc(red, test_set)/test_size)*100)
    plt.plot(results)
    plt.ylabel('accuracy')
    plt.xlabel('epochs')
    plt.show()

main()
