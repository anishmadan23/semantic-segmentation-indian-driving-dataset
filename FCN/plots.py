import matplotlib.pyplot as plt
import numpy as np

f = open('log.txt', 'r')
lines = f.readlines()
train_loss = []
val_loss = []
i = -1
for l in lines:
	if(l[0]=='L'):
		s = l.split()[-1]
		if(i==-1):
			train_loss.append(float(s))
		else:
			val_loss.append(float(s))
		i*=-1

plt.plot(train_loss, color='r', label = 'training')
plt.plot(val_loss, color='b', label = 'validation')
plt.legend()
plt.show()