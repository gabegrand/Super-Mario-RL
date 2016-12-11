import matplotlib.pyplot as plt
import fnmatch
import os
import pickle
import util

xs = []
horzVelocity = []
rightAction = []
gapRightFar = []

distance = []

for i in xrange(10, 250, 10):
	string = '*iter-' + str(i) + '.pickle'
	for file in os.listdir('plot'):
		if fnmatch.fnmatch(file, string):
			print "Opening: " + string
			with open('plot/' + file) as handle:
				xs.append(i)

				save_dict = pickle.load(handle)

				weights = save_dict['weights']
				horzVelocity.append(weights['horzVelocity'])
				rightAction.append(weights['rightAction'])
				gapRightFar.append(weights['gapRightFar'])

				diagnostics = save_dict['diagnostics']
				distance.append(diagnostics['distance'])

_, ax = plt.subplots(1, 3, figsize=(24, 8))

ax[0].plot(xs, horzVelocity)
ax[0].set_ylabel('horzVelocity')

ax[1].plot(xs, rightAction)
ax[1].set_ylabel('rightAction')

ax[2].plot(xs, gapRightFar)
ax[2].set_ylabel('gapRightFar')
plt.show()

plt.plot(xs, distance)
plt.show()
