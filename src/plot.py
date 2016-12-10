import matplotlib.pyplot as plt
import fnmatch
import os
import pickle
import util

xs = []
movingRight = []
canMoveLeft = []
groundBelow = []
for i in xrange(10, 1150, 100):
	string = '*iter-' + str(i) + '.pickle'
	for file in os.listdir('save'):
		if fnmatch.fnmatch(file, string):
			with open('save/' + file) as handle:
				weights = pickle.load(handle)['weights']
				xs.append(i)
		        movingRight.append(weights['movingRight'])
		        canMoveLeft.append(weights['canMoveLeft'])
		        groundBelow.append(weights['groundBelow'])

_, ax = plt.subplots(1, 3, figsize=(24, 8))

ax[0].plot(xs, movingRight)
ax[0].set_ylabel('movingRight')

ax[1].plot(xs, canMoveLeft)
ax[1].set_ylabel('canMoveLeft')

ax[2].plot(xs, groundBelow)
ax[2].set_ylabel('groundBelow')
plt.show()