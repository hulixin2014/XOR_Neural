# coding-utf-8
import neural
import numpy as np
x = np.array([[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
y = np.array([0, 1, 1, 0])
w1=[0.1,-0.2,0.2]
w2=[0.1,0.2,-0.2]
w3=[0.1,-0.3,-0.3]
# w1=[1,-1,1]
# w2=[1,0,-1]
# w3=[1,0,-1]
# w1=[0,-1,1]
# w2=[0,0,-1]
# w3=[0,0,-1]
w1,w2,w3=neural.train(w1,w2,w3,x,y,0.4,0)
for idx in x:
    print neural.predict(w1,w2,w3,idx)
