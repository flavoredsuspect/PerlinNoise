import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from Clases import Data, Menu
import numpy as np
from numpy import savetxt
import matplotlib.pyplot as plt

## WARNING = methos copies the pointer to the data cell!!!! WARNING##
"""
Menu()
input()
"""
Bar = list()
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
model = load_model('C:/Users/anpag/Desktop/CODE/mnist.h5')

mask = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]])
a = list()
a.append(mask)
index = list()
d = list(y_test)

p=7
indices = [i for i, x in enumerate(d) if x == p]

for ll in range(0, 100):
    index.append(indices[ll])

for i in index:
    b = Data(x_test[i].reshape(28, 28), model)
    Bar.append(b.spoil(1, a, False))

Res = sum(Bar)
Res = Res / len(index)
Results = Res
name = 'results{number}.csv'
name = name.format(number=str(p))
savetxt(name, Results, delimiter=',')
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from Clases import Data, Menu
import numpy as np
from numpy import savetxt
import matplotlib.pyplot as plt

## WARNING = methos copies the pointer to the data cell!!!! WARNING##
"""
Menu()
input()
"""
Bar = list()
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
model = load_model('C:/Users/anpag/Desktop/CODE/mnist.h5')

mask = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]])
a = list()
a.append(mask)
index = list()
d = list(y_test)

for pp in range(0, 10):
    p=9-pp
    indices = [i for i, x in enumerate(d) if x == p]

    for ll in range(0, 100):
        index.append(indices[ll])

    for i in index:
        b = Data(x_test[i].reshape(28, 28), model)
        Bar.append(b.spoil(1, a, False))

    Res = sum(Bar)
    Res = Res / len(index)
    Results = Res
    name = 'results{number}.csv'
    name = name.format(number=str(p))
    savetxt(name, Results, delimiter=',')

# data = loadtxt('data.csv', delimiter=',')

"""
    ONE NUMBER CHECKING

model = load_model('C:/Users/apari/Desktop/CODE/mnist.h5')
I=np.ones((28,28))
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
b=Data(x_test[2853].reshape(28, 28),model)
mask=np.array([[0, 1, 1],[0, 0, 1],[0, 0, 0]])
a=list()
a.append(mask)
Bar=b.spoil(1,a,False)

plt.imshow(255*Bar,cmap='Greys')
"""

"""
model = load_model('C:/Users/apari/Desktop/CODE/mnist.h5')

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

image_index = 2853
plt.imshow(x_test[image_index].reshape(28, 28), cmap='Greys')
print(x_test[image_index].reshape(1, 28, 28, 1))
predict = x_test[image_index].reshape(28,28)
pred = model.predict(x_test[image_index].reshape(1, 28, 28, 1))
print(pred.argmax())

        for i in range(0,len(c)):
    p=np.where(c[i]==1,255,0)
    plt.imshow(p,cmap='Greys')
    plt.pause(3)
    o=input()

"""



# data = loadtxt('data.csv', delimiter=',')

"""
    ONE NUMBER CHECKING

model = load_model('C:/Users/apari/Desktop/CODE/mnist.h5')
I=np.ones((28,28))
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
b=Data(x_test[2853].reshape(28, 28),model)
mask=np.array([[0, 1, 1],[0, 0, 1],[0, 0, 0]])
a=list()
a.append(mask)
Bar=b.spoil(1,a,False)

plt.imshow(255*Bar,cmap='Greys')
"""

"""
model = load_model('C:/Users/apari/Desktop/CODE/mnist.h5')

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

image_index = 2853
plt.imshow(x_test[image_index].reshape(28, 28), cmap='Greys')
print(x_test[image_index].reshape(1, 28, 28, 1))
predict = x_test[image_index].reshape(28,28)
pred = model.predict(x_test[image_index].reshape(1, 28, 28, 1))
print(pred.argmax())

        for i in range(0,len(c)):
    p=np.where(c[i]==1,255,0)
    plt.imshow(p,cmap='Greys')
    plt.pause(3)
    o=input()

"""


