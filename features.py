import hyperparameters as hp
import numpy as np
import sys
import util

# Returns a vector (actually a util.Counter object) of features
def getFeatures(state):
	assert isinstance(state, util.State)

	curr_state = state.getCurr()
	prev_state = state.getPrev()

	curr_dist = state.currDist
	prev_dist = state.prevDist

	# Get Mario's position
	curr_mpos = marioPosition(curr_state)

	# Make sure Mario is currently on the screen
	if not curr_mpos:
		raise ValueError("getFeatures: curr_mpos is None")

	features = util.Counter()

	features['canMoveLeft'] = canMoveLeft(curr_state)
	features['canMoveRight'] = canMoveRight(curr_state)
	features['canMoveUp'] = canMoveUp(curr_state)
	features['canMoveDown'] = canMoveDown(curr_state)

	features['horzVelocity'] = horzVelocity(prev_dist, curr_dist)

	# features['movingUp'] = movingUp(prev_state, curr_state)
	# features['movingDown'] = movingDown(prev_state, curr_state)
	features['groundVertDistance'] = groundVertDistance(curr_state)
	features['roofVertDistance'] = roofVertDistance(curr_state)
	# features['groundLeftDistance'] = groundLeftDistance(curr_state)
	# features['groundRightDistance'] = groundRightDistance(curr_state)
	# features['distLeftEnemy'] = distLeftEnemy(curr_state)
	# features['distRightEnemy'] = distRightEnemy(curr_state)
	# features['distUpEnemy'] = distUpEnemy(curr_state)
	# features['distDownEnemy'] = distDownEnemy(curr_state)
	features['enemyOnScreen'] = enemyOnScreen(curr_state)
	features['groundBelow'] = groundBelow(curr_state)

	return features

# Returns mario's position as row, col pair
# Returns None if Mario not on map
# Always perform None check on return val
# For functions in this file, use _marioPosition, since functions
# should only be called if mario is on screen
def marioPosition(state):
	if state is None:
		return None
	rows, cols = np.nonzero(state == 3)
	if rows.size == 0 or cols.size == 0:
		if hp.DISPLAY_WARNINGS:
			print "WARNING: Mario is off the map"
		return None
	else:
		return rows[0], cols[0]

# FEATURE FUNCTIONS --- Any features used should be scaled [0,1]

# Binary feature as to whether Mario is moving up
# Only call if mario is currently on screen
def movingUp(prev_state, curr_state):
	prev_mpos = marioPosition(prev_state)
	curr_mpos = marioPosition(curr_state)
	assert curr_mpos is not None
	if prev_mpos is None:
		return 0.0
	prev_row, _ = prev_mpos
	curr_row, _ = curr_mpos

	if curr_row < prev_row:
		return 1.0
	else:
		return 0.0

# Binary feature as to whether Mario is moving down
# Only call if mario is currently on screen
def movingDown(prev_state, curr_state):
	prev_mpos = marioPosition(prev_state)
	curr_mpos = marioPosition(curr_state)
	assert curr_mpos is not None
	if prev_mpos is None:
		return 0.0
	prev_row, _ = prev_mpos
	curr_row, _ = curr_mpos

	if curr_row > prev_row:
		return 1.0
	else:
		return 0.0

# Mario's velocity on the x axis.
# Scaled to [0, 1], since max velocity is 8.0
def horzVelocity(prev_dist, curr_dist):
	if prev_dist and curr_dist:
		return float(curr_dist - prev_dist) / 8.0
	else:
		return 0.0

# Returns the number of rows between Mario and the roof (0 if next level is roof)
# if no roof above Mario, return number of rows between Mario and offscreen
# Only call if Mario is on screen
# Norm factor is state.shape[0], which is 13
def roofVertDistance(state):
	m_row, m_col = _marioPosition(state)

	# get the rows in Mario's column with objects, if any
	col_contents = state[:m_row, m_col]
	obj_rows = np.nonzero(col_contents == 1)

	if obj_rows[0].size == 0:
		return float(m_row) / state.shape[0]
	else:
		return float(m_row - obj_rows[0][-1] - 1) / state.shape[0]

# Returns the vert distance from Mario to the ground, 0 if on ground
# if no ground below Mario, return number of rows to offscreen
# Only call if Mario is on screen
# Norm factor is state.shape[0], which is 13
def groundVertDistance(state):
	m_row, m_col = _marioPosition(state)

	if m_row < state.shape[0] - 1:
		# get the rows in Mario's column with objects, if any
		col_contents = state[m_row + 1:, m_col]
		obj_vert_dists = np.nonzero(col_contents == 1)

		if obj_vert_dists[0].size == 0:
			return float(state.shape[0] - m_row - 1) / state.shape[0]
		else:
			return float(obj_vert_dists[0][0]) / state.shape[0]
	else:
		return 1.0 / state.shape[0]

# Returns the # of columns to the left of Mario for which
# there exists at least one object (ground=1) at a height lower than Mario
# Only call if Mario is on screen
# Norm factor is state.shape[1] - 1, which is 15
def groundLeftDistance(state):
	m_row, m_col = _marioPosition(state)

	# if Mario at bottom of screen or in left most column, no ground to left
	if m_row == state.shape[0] - 1 or m_col == 0:
		return 0.0

	norm = state.shape[1] - 1.0
	dist = 0.0

	for col in xrange(m_col - 1, -1, -1):
		col_contents = state[m_row + 1:, col]
		obj_vert_dists = np.nonzero(col_contents == 1)

		if obj_vert_dists[0].size == 0:
			return dist / norm
		else:
			dist += 1.0

	return dist / norm

# Returns the # of columns to the right of Mario for which
# there exists at least one object (ground=1) at a height lower than Mario
# Only call if Mario is on screen
# Norm factor is state.shape[1] - 1, which is 15
def groundRightDistance(state):
	m_row, m_col = _marioPosition(state)

	# if Mario at bottom of screen or in right most column, no ground to right
	if m_row == state.shape[0] - 1 or m_col == state.shape[1] - 1:
		return 0.0

	norm = state.shape[1] - 1.0
	dist = 0.0

	for col in xrange(m_col + 1, state.shape[1]):
		col_contents = state[m_row + 1:, col]
		obj_vert_dists = np.nonzero(col_contents == 1)

		if obj_vert_dists[0].size == 0:
			return dist / norm
		dist += 1.0

	return dist / norm

# left distance to nearest enemy (dist to edge of screen if no enemy)
# Only call if Mario is on screen
# Norm factor is state.shape[1], which is 16
def distLeftEnemy(state):
	m_row, m_col = _marioPosition(state)

	# if no enemies, return left horizontal distance remaining on screen
	e_rows, e_cols = np.nonzero(state[:, :m_col + 1] == 2)
	if e_rows.size == 0 or e_cols.size == 0:
		return float(m_col + 1) / state.shape[1]

	# find the nearest enemy
	nearest = None

	# look to the left
	for col_num in xrange(m_col, -1, -1):
		col_contents = state[:, col_num]
		enemies = np.nonzero(col_contents == 2)
		# if we've found an enemy, we're as close as we'll get
		if enemies[0].size > 0:
			nearest = m_col - col_num
			break

	return float(nearest) / state.shape[1]

# right distance to nearest enemy (dist to edge of screen if no enemy)
# Only call if Mario is on screen
# Norm factor is state.shape[1], which is 16
def distRightEnemy(state):
	m_row, m_col = _marioPosition(state)

	# if no enemies, return right horizontal distance remaining on screen
	e_rows, e_cols = np.nonzero(state[:, m_col:] == 2)
	if e_rows.size == 0 or e_cols.size == 0:
		return float(state.shape[1] - m_col) / state.shape[1]

	# find the nearest enemy
	nearest = None

	# look to the right
	for col_num in xrange(m_col, state.shape[1]):
		col_contents = state[:, col_num]
		enemies = np.nonzero(col_contents == 2)
		if enemies[0].size > 0:
			nearest = col_num - m_col
			break

	return float(nearest) / state.shape[1]

# up distance to nearest enemy (dist to edge of screen if no enemy)
# Only call if Mario is on screen
# Norm factor is state.shape[0], which is 13
def distUpEnemy(state):
	m_row, m_col = _marioPosition(state)

	# if no enemies, return up vert dist to edge
	e_rows, e_cols = np.nonzero(state[:m_row + 1, :] == 2)
	if e_rows.size == 0 or e_cols.size == 0:
		return float(m_row + 1) / state.shape[0]

	# find the nearest enemy
	nearest = None

	# look above
	for row_num in xrange(m_row, -1, -1):
		row_contents = state[row_num, :]
		enemies = np.nonzero(row_contents == 2)

		# if we've found an enemy, we're as close as we'll get
		if enemies[0].size > 0:
			nearest = m_row - row_num
			break

	return float(nearest) / state.shape[0]

# down distance to nearest enemy (dist to edge of screen if no enemy)
# Only call if Mario is on screen
# Norm factor is state.shape[0], which is 13
def distDownEnemy(state):
	m_row, m_col = _marioPosition(state)

	# if no enemies, return down vert dist to edge
	e_rows, e_cols = np.nonzero(state[m_row:, :] == 2)
	if e_rows.size == 0 or e_cols.size == 0:
		return float(state.shape[0] - m_row) / state.shape[0]

	# find the nearest enemy
	nearest = None

	# look below
	for row_num in xrange(m_row, state.shape[0]):
		row_contents = state[row_num, :]
		enemies = np.nonzero(row_contents == 2)
		if enemies[0].size > 0:
			nearest = row_num - m_row
			break

	return float(nearest) / state.shape[0]

# Return whether there is one or more enemy on screen (1=true)
def enemyOnScreen(state):
	rows, cols = np.nonzero(state == 2)
	if rows.size == 0 or cols.size == 0:
		return 0.0
	return 1.0

# Return whether there is ground below Mario (1=true)
# Only call if Mario is on screen
def groundBelow(state):
	m_row, m_col = _marioPosition(state)

	# get the rows in Mario's column with objects, if any
	col_contents = state[m_row:, m_col]
	ground_below = np.nonzero(col_contents == 1)

	if ground_below[0].size == 0:
		return 0.0
	return 1.0

# Return whether Mario can move up in his position (1=true)
# Only call if Mario is on screen
def canMoveUp(state):
	m_row, m_col = _marioPosition(state)

	if m_row > 0:
		if state[m_row-1, m_col] in [0, 3]:
			return 1.0
		return 0.0
	return 1.0

# Return whether Mario can move down in his position (1=true)
# Only call if Mario is on screen
def canMoveDown(state):
	m_row, m_col = _marioPosition(state)

	if m_row < state.shape[0] - 1:
		if state[m_row+1, m_col] in [0, 3]:
			return 1.0
		return 0.0
	return 1.0

# Return whether Mario can move right in his position (1=true)
# Only call if Mario is on screen
def canMoveRight(state):
	m_row, m_col = _marioPosition(state)

	if m_col < state.shape[1] - 1:
		if state[m_row, m_col + 1] in [0, 3]:
			return 1.0
		return 0.0
	return 1.0

# Return whether Mario can move left in his position (1=true)
# Only call if Mario is on screen
def canMoveLeft(state):
	m_row, m_col = _marioPosition(state)

	if m_col > 0:
		if state[m_row, m_col - 1] in [0, 3]:
			return 1.0
		return 0.0
	return 0.0

# PRIVATE FUNCTIONS

# version of marioPosition that fails if Mario not on screen
def _marioPosition(state):
	rows, cols = np.nonzero(state == 3)
	assert rows.size != 0 and cols.size != 0
	return rows[0], cols[0]
