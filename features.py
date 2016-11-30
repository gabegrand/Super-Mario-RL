import numpy as np
import sys
import util

# Returns a vector (actually a util.Counter object) of features
def getFeatures(state, action):

	# Get Mario's position
	# prev_mpos = marioPosition(prev_state)
	curr_mpos = marioPosition(state)

	# Make sure Mario is on the screen
	# if not (prev_mpos and curr_mpos):
	if not curr_mpos:
		raise ValueError("getFeatures: Mario position is None")

	features = util.Counter()

	features['canMoveLeft'] = canMoveLeft(state)
	features['canMoveRight'] = canMoveRight(state)
	features['canMoveUp'] = canMoveLeft(state)
	features['canMoveDown'] = canMoveLeft(state)
	# features['movingUp'] = movingUp(prev_state, state, action)
	# features['movingDown'] = movingDown(prev_state, state, action)
	# features['movingLeft'] = movingLeft(prev_state, state, action)
	# features['movingRight'] = movingRight(prev_state, state, action)
	features['groundVertDistance'] = groundVertDistance(state)
	features['roofVertDistance'] = roofVertDistance(state)
	features['groundLeftDistance'] = groundLeftDistance(state)
	features['groundRightDistance'] = groundRightDistance(state)
	features['distLeftEnemy'] = distLeftEnemy(state)
	features['distRightEnemy'] = distRightEnemy(state)
	features['distUpEnemy'] = distUpEnemy(state)
	features['distDownEnemy'] = distDownEnemy(state)
	features['enemyOnScreen'] = enemyOnScreen(state)
	features['groundBelow'] = groundBelow(state)

	return features

# Returns mario's position as row, col pair
# Returns None if Mario not on map
# Always perform None check on return val
# For functions in this file, use _marioPosition, since functions
# should only be called if mario is on screen
def marioPosition(state):
	rows, cols = np.nonzero(state == 3)
	if rows.size == 0 or cols.size == 0:
		print "WARNING: Mario is off the map"
		return None
	else:
		return rows[0], cols[0]

# FEATURE FUNCTIONS --- Any features used should be scaled [0,1]

# Binary feature as to whether Mario is moving up
# Only call if mario is currently on screen
def movingUp(prev_state, curr_state, action_num):
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
def movingDown(prev_state, curr_state, action_num):
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

# Binary feature as to whether Mario is moving left
# Only call if mario is currently on screen
def movingLeft(prev_state, curr_state, action_num):
	prev_mpos = marioPosition(prev_state)
	curr_mpos = marioPosition(curr_state)
	assert curr_mpos is not None
	if prev_mpos is None:
		return 0.0
	_, prev_col = prev_mpos
	_, curr_col = curr_mpos

	if curr_col < prev_col:
		return 1.0
	else:
		return 0.0

# Binary feature as to whether Mario is moving right
# Only call if mario is currently on screen
def movingRight(prev_state, curr_state, action_num):
	prev_mpos = marioPosition(prev_state)
	curr_mpos = marioPosition(curr_state)
	assert curr_mpos is not None
	if prev_mpos is None:
		return 0.0
	_, prev_col = prev_mpos
	_, curr_col = curr_mpos

	if curr_col > prev_col or (action_num in [7,8,9,10] and canMoveRight(prev_state)):
		return 1.0
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
		if state[m_row-1, m_col] == 0:
			return 1.0
		return 0.0
	return 1.0

# Return whether Mario can move down in his position (1=true)
# Only call if Mario is on screen
def canMoveDown(state):
	m_row, m_col = _marioPosition(state)

	if m_row < state.shape[0] - 1:
		if state[m_row+1, m_col] == 0:
			return 1.0
		return 0.0
	return 1.0

# Return whether Mario can move right in his position (1=true)
# Only call if Mario is on screen
def canMoveRight(state):
	m_row, m_col = _marioPosition(state)

	if m_col < state.shape[1] - 1:
		if state[m_row, m_col + 1] == 0:
			return 1.0
		return 0.0
	return 1.0

# Return whether Mario can move left in his position (1=true)
# Only call if Mario is on screen
def canMoveLeft(state):
	m_row, m_col = _marioPosition(state)

	if m_col > 0:
		if state[m_row, m_col - 1] == 0:
			return 1.0
		return 0.0
	return 0.0

# PRIVATE FUNCTIONS

# version of marioPosition that fails if Mario not on screen
def _marioPosition(state):
	rows, cols = np.nonzero(state == 3)
	assert rows.size != 0 and cols.size != 0
	return rows[0], cols[0]
