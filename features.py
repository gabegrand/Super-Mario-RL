import numpy as np
import sys


# Returns a vector (actually a util.Counter object) of features
def getFeatures(prev_state, state, info):

	# Get Mario's position
	prev_mpos = marioPosition(prev_state)
	curr_mpos = marioPosition(state)

	# Make sure Mario is on the screen
	if not (prev_state and state):
		print "getFeatures: Mario position is None"
		return None

	features = util.Counter()

	features['movingUp'] = None
	features['movingDown'] = None
	features['movingLeft'] = None
	features['movingRight'] = None
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
	features['canMoveLeft'] = canMoveLeft(state)
	features['canMoveRight'] = canMoveRight(state)
	features['canMoveUp'] = canMoveLeft(state)
	features['canMoveDown'] = canMoveLeft(state)
	features['marioStatus'] = marioStatus(info)

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
	return rows[0], cols[0]

# Returns a number corresponding to Mario's direction
# 0 - stationary, 1 - up, 2 - up/right
# 3 - right, 4 - down/right, 5 - down
# 6 - down/left, 7 - left, 8 - up/left
# TODO what about case where mario moves from right edge of screen to left?
# is this possible?
# TODO direction assignment might not be linear, talk to Gabe
# TODO screen moves, so moving a column to the right doesn't necessarily mean right
def marioDirection(prev_mpos, curr_mpos):
	assert curr_mpos is not None
	if prev_mpos is None:
		return 0
	prev_row, prev_col = prev_mpos
	curr_row, curr_col = curr_mpos

	if curr_row == prev_row:
		# stationary
		if curr_col == prev_col:
			return 0
		# right
		elif curr_col > prev_col:
			return 3
		# left (curr_col < prev_col)
		else:
			return 7

	elif curr_row > prev_row:
		# down
		if curr_col == prev_col:
			return 5
		# down-right
		elif curr_col > prev_col:
			return 4
		# down-left (curr_col < prev_col)
		else:
			return 6

	# curr_row < prev_row
	else:
		# up
		if curr_col == prev_col:
			return 1
		# up-right
		elif curr_col > prev_col:
			return 2
		# up-left (curr_col < prev_col)
		else:
			return 8

# Returns the vert distance from Mario to the ground
# if no ground below Mario, return number of rows to offscreen
# Only call if Mario is on screen
def groundVertDistance(state):
	m_row, m_col = _marioPosition(state)

	# get the rows in Mario's column with objects, if any
	col_contents = state[m_row:, m_col]
	obj_vert_dists = np.nonzero(col_contents == 1)

	if obj_vert_dists[0].size == 0:
		return state.shape[0] - m_row
	return obj_vert_dists[0][0]

# Returns the number of rows between Mario and the roof (0 if next level is roof)
# if no roof above Mario, return number of rows between Mario and offscreen
# Only call if Mario is on screen
def roofVertDistance(state):
	m_row, m_col = _marioPosition(state)

	# get the rows in Mario's column with objects, if any
	col_contents = state[:m_row, m_col]
	obj_vert_dists = np.nonzero(col_contents == 1)

	if obj_vert_dists[0].size == 0:
		return m_row + 1
	return m_row - obj_vert_dists[0][-1]

# Returns the # of columns to the right of Mario for which
# there exists at least one object (ground=1) at a height lower than Mario
# Only call if Mario is on screen
def groundRightDistance(state):
	m_row, m_col = _marioPosition(state)

	dist = 0

	if m_row < state.shape[0] - 1 and m_col < state.shape[1] - 1:
		for col in xrange(m_col + 1, state.shape[1]):
			col_contents = state[m_row + 1:, col]
			obj_vert_dists = np.nonzero(col_contents == 1)

			if obj_vert_dists[0].size == 0:
				return dist
			dist += 1
	return dist

# Returns the # of columns to the left of Mario for which
# there exists at least one object (ground=1) at a height lower than Mario
# Only call if Mario is on screen
def groundLeftDistance(state):
	m_row, m_col = _marioPosition(state)

	dist = 0

	if m_row < state.shape[0] - 1 and m_col > 0:
		for col in xrange(m_col - 1, -1, -1):
			col_contents = state[m_row + 1:, col]
			obj_vert_dists = np.nonzero(col_contents == 1)

			if obj_vert_dists[0].size == 0:
				return dist
			dist += 1
	return dist


# left distance to nearest enemy (dist to edge of screen if no enemy)
# Only call if Mario is on screen
def distLeftEnemy(state):
	m_row, m_col = _marioPosition(state)

	# if no enemies, return left horizontal distance remaining on screen
	e_rows, e_cols = np.nonzero(state[:, :m_col + 1] == 2)
	if e_rows.size == 0 or e_cols.size == 0:
		return m_col + 1

	# find the nearest enemy
	nearest = sys.maxsize

	# look to the left
	for col_num in xrange(m_col, -1, -1):
		col_contents = state[:, col_num]
		enemies = np.nonzero(col_contents == 2)
		# if we've found an enemy, we're as close as we'll get
		if enemies[0].size > 0:
			nearest = m_col - col_num
			break

	return nearest

# right distance to nearest enemy (dist to edge of screen if no enemy)
# Only call if Mario is on screen
def distRightEnemy(state):
	m_row, m_col = _marioPosition(state)

	# if no enemies, return right horizontal distance remaining on screen
	e_rows, e_cols = np.nonzero(state[:, m_col:] == 2)
	if e_rows.size == 0 or e_cols.size == 0:
		return state.shape[1] - m_col

	# find the nearest enemy
	nearest = sys.maxsize

	# look to the right
	for col_num in xrange(m_col, state.shape[1]):
		col_contents = state[:, col_num]
		enemies = np.nonzero(col_contents == 2)
		if enemies[0].size > 0:
			nearest = col_num - m_col
			break

	return nearest

# up distance to nearest enemy (dist to edge of screen if no enemy)
# Only call if Mario is on screen
def distUpEnemy(state):
	m_row, m_col = _marioPosition(state)

	# if no enemies, return up vert dist to edge
	e_rows, e_cols = np.nonzero(state[:m_row + 1, :] == 2)
	if e_rows.size == 0 or e_cols.size == 0:
		return m_row + 1

	# find the nearest enemy
	nearest = sys.maxsize

	# look above
	for row_num in xrange(m_row, -1, -1):
		row_contents = state[row_num, :]
		enemies = np.nonzero(row_contents == 2)

		# if we've found an enemy, we're as close as we'll get
		if enemies[0].size > 0:
			nearest = m_row - row_num
			break

	return nearest

# down distance to nearest enemy (dist to edge of screen if no enemy)
# Only call if Mario is on screen
def distDownEnemy(state):
	m_row, m_col = _marioPosition(state)

	# if no enemies, return down vert dist to edge
	e_rows, e_cols = np.nonzero(state[m_row:, :] == 2)
	if e_rows.size == 0 or e_cols.size == 0:
		return state.shape[0] - m_row

	# find the nearest enemy
	nearest = sys.maxsize

	# look below
	for row_num in xrange(m_row, state.shape[0]):
		row_contents = state[row_num, :]
		enemies = np.nonzero(row_contents == 2)
		if enemies[0].size > 0:
			nearest = row_num - m_row
			break

	return nearest

# Return whether there is one or more enemy on screen (1=true)
def enemyOnScreen(state):
	rows, cols = np.nonzero(state == 2)
	if rows.size == 0 or cols.size == 0:
		return 0
	return 1

# Return whether there is ground below Mario (1=true)
# Only call if Mario is on screen
def groundBelow(state):
	m_row, m_col = _marioPosition(state)

	# get the rows in Mario's column with objects, if any
	col_contents = state[m_row:, m_col]
	ground_below = np.nonzero(col_contents == 1)

	if ground_below[0].size == 0:
		return 0
	return 1

# Return whether Mario can jump in his position (1=true)
# Only call if Mario is on screen
def canJump(state):
	m_row, m_col = _marioPosition(state)

	if m_row > 0:
		if state[m_row-1, m_col] == 0:
			return 1
		return 0
	return 1

# Return whether Mario can move right in his position (1=true)
# Only call if Mario is on screen
def canMoveRight(state):
	m_row, m_col = _marioPosition(state)

	if m_col < state.shape[1] - 1:
		if state[m_row, m_col + 1] == 0:
			return 1
		return 0
	return 1

# Return whether Mario can move left in his position (1=true)
# Only call if Mario is on screen
def canMoveLeft(state):
	m_row, m_col = _marioPosition(state)

	if m_col > 0:
		if state[m_row, m_col - 1] == 0:
			return 1
		return 0
	return 0

# Return the status of mario
# 0 = small, 1 = big, 2+ = fireball
def marioStatus(info):
	return _fetchEntry(info, "player_status")

# Return the number of seconds remaining in the level
def timeRemaining(info):
	return _fetchEntry(info, "time")

# PRIVATE FUNCTIONS

# Return the value for key in info dict, printing errors where applicable
def _fetchEntry(info, key):
	if key in info.keys():
		if info[key] == -1:
			print "WARNING: " + key + " unknown in info dict"
		return info[key]
	print "WARNING: " + key + " not in info dict"
	return None

# version of marioPosition that fails if Mario not on screen
def _marioPosition(state):
	rows, cols = np.nonzero(state == 3)
	assert rows.size != 0 and cols.size != 0
	return rows[0], cols[0]
