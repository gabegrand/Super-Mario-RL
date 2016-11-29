import numpy as np
import sys

# Returns mario's position as row, col pair
# Returns None if Mario not on map
# Always perform None check on return val
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
# TODO what about case where mario moves off screen? is this possible?
# TODO direction assignment might not be linear, talk to Gabe
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

def _test_bounds(state, left, right, above, below):
	assert groundLeftDistance(state) == left
	assert groundRightDistance(state) == right
	assert roofVertDistance(state) == above
	assert groundVertDistance(state) == below

def _test_enemy_dists(state, dLeft, dRight, dUp, dDown):
	assert distLeftEnemy(state) == dLeft
	assert distRightEnemy(state) == dRight
	assert distUpEnemy(state) == dUp
	assert distDownEnemy(state) == dDown

def _testEnemyOnScreen():
	print "Testing enemy on screen"
	a = np.array([[1,1,1], [3,1,0], [0,2,1]])
	b = np.array([[1,2,1], [0,3,0], [1,1,2]])
	c = np.array([[1,3,1], [0,0,0], [1,1,1]])
	d = np.array([[3,0,2], [0,0,1], [0,1,1]])
	e = np.array([[1,0,1], [0,0,1], [2,2,3]])

	assert enemyOnScreen(a) == 1
	assert enemyOnScreen(b) == 1
	assert enemyOnScreen(c) == 0
	assert enemyOnScreen(d) == 1
	assert enemyOnScreen(e) == 1
	print "All tests passed!"


def _testCanMoveLeft():
	print "Testing can move left"
	a = np.array([[1,1,1], [3,0,0], [1,1,1]])
	b = np.array([[1,1,1], [0,3,0], [1,1,1]])
	c = np.array([[1,1,1], [0,1,3], [1,1,1]])

	assert canMoveLeft(a) == 0
	assert canMoveLeft(b) == 1
	assert canMoveLeft(c) == 0
	print "All tests passed!"


def _testCanMoveRight():
	print "Testing can move right"
	a = np.array([[1,1,1], [3,1,0], [1,1,1]])
	b = np.array([[1,1,1], [0,3,0], [1,1,1]])
	c = np.array([[1,1,1], [0,1,3], [1,1,1]])

	assert canMoveRight(a) == 0
	assert canMoveRight(b) == 1
	assert canMoveRight(c) == 1
	print "All tests passed!"

def _testCanJump():
	print "Testing can jump"
	a = np.array([[1,1,1], [3,1,0], [1,1,1]])
	b = np.array([[1,0,1], [0,3,0], [1,1,1]])
	c = np.array([[1,1,1], [0,1,0], [1,1,3]])
	d = np.array([[1,0,1], [0,0,1], [1,1,3]])
	e = np.array([[1,3,1], [0,0,1], [1,1,1]])

	assert canJump(a) == 0
	assert canJump(b) == 1
	assert canJump(c) == 1
	assert canJump(d) == 0
	assert canJump(e) == 1
	print "All tests passed!"

def _testGroundBelow():
	print "Testing ground below"
	a = np.array([[1,1,1], [3,1,0], [0,1,1]])
	b = np.array([[1,0,1], [0,3,0], [1,1,1]])
	c = np.array([[1,3,1], [0,0,0], [1,1,1]])
	d = np.array([[3,0,1], [0,0,1], [0,1,1]])
	e = np.array([[1,0,1], [0,0,1], [0,1,3]])

	assert groundBelow(a) == 0
	assert groundBelow(b) == 1
	assert groundBelow(c) == 1
	assert groundBelow(d) == 0
	assert groundBelow(e) == 0
	print "All tests passed!"

def main():
	print "Running boundary tests"
	a = np.array([[1,1,1], [3,0,0], [1,1,1]])
	b = np.array([[1,1,1], [0,3,0], [1,1,1]])
	c = np.array([[1,1,1], [0,0,3], [1,1,1]])

	_test_bounds(a, 0, 2, 1, 1)
	_test_bounds(b, 1, 1, 1, 1)
	_test_bounds(c, 2, 0, 1, 1)

	a = np.array([[1,1,1], [3,0,0], [0,0,0]])
	b = np.array([[1,1,1], [0,3,0], [0,0,0]])
	c = np.array([[1,1,1], [0,0,3], [0,0,0]])

	_test_bounds(a, 0, 0, 1, 2)
	_test_bounds(b, 0, 0, 1, 2)
	_test_bounds(c, 0, 0, 1, 2)

	a = np.array([[0,0,0], [3,0,0], [1,1,1]])
	b = np.array([[0,0,0], [0,3,0], [1,1,1]])
	c = np.array([[0,0,0], [0,0,3], [1,1,1]])

	_test_bounds(a, 0, 2, 2, 1)
	_test_bounds(b, 1, 1, 2, 1)
	_test_bounds(c, 2, 0, 2, 1)

	a = np.array([[1,0,1], [3,0,0], [1,0,1]])
	b = np.array([[1,0,1], [0,3,0], [1,0,1]])
	c = np.array([[1,0,1], [0,0,3], [1,0,1]])

	_test_bounds(a, 0, 0, 1, 1)
	_test_bounds(b, 1, 1, 2, 2)
	_test_bounds(c, 0, 0, 1, 1)

	a = np.array([[0,0,0,0,0], [0,0,0,0,0], [0,1,3,1,0], [0,0,0,1,0], [1,1,0,0,0]])
	b = np.array([[0,1,0,0,0], [1,3,1,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,1,1,1,1]])
	c = np.array([[0,0,0,1,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,1,3,1], [0,1,1,0,0]])

	_test_bounds(a, 2, 1, 3, 3)
	_test_bounds(b, 0, 3, 1, 3)
	_test_bounds(c, 2, 0, 3, 2)

	print "All tests passed!"

	print "Running enemy distance tests"

	a = np.array([[1,1,1], [3,0,0], [1,1,1]])
	b = np.array([[1,1,1], [0,3,0], [1,1,1]])
	c = np.array([[1,1,1], [1,1,1], [0,0,3]])

	_test_enemy_dists(a, 1, 3, 2, 2)
	_test_enemy_dists(b, 2, 2, 2, 2)
	_test_enemy_dists(c, 3, 1, 3, 1)

	a = np.array([[1,1,1], [3,0,0], [0,0,2]])
	b = np.array([[0,2,0], [0,3,0], [1,1,1]])
	c = np.array([[1,1,1], [0,0,3], [2,0,0]])

	_test_enemy_dists(a, 1, 2, 2, 1)
	_test_enemy_dists(b, 0, 0, 1, 2)
	_test_enemy_dists(c, 2, 1, 2, 1)

	a = np.array([[1,2,0], [1,3,0], [1,2,1]])
	b = np.array([[0,0,0], [2,3,0], [2,1,1]])
	c = np.array([[0,0,1], [2,3,2], [1,1,1]])

	_test_enemy_dists(a, 0, 0, 1, 1)
	_test_enemy_dists(b, 1, 2, 0, 0)
	_test_enemy_dists(c, 1, 1, 0, 0)

	print "All tests passed!"

	_testCanJump()
	_testCanMoveLeft()
	_testCanMoveRight()
	_testEnemyOnScreen()
	_testGroundBelow()

	print "End of tests"

if __name__ == "__main__": main()
