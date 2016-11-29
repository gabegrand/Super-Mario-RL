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

# Returns the vert distance from Mario to the ground
# if no ground below Mario, return number of rows to offscreen
# Return None if Mario not on screen
# Always perform None check on return val
def groundVertDistance(state):
	mario_pos = marioPosition(state)
	if mario_pos is None:
		return None
	m_row, m_col = mario_pos

	# get the rows in Mario's column with objects, if any
	col_contents = state[m_row:, m_col]
	obj_vert_dists = np.nonzero(col_contents == 1)

	if obj_vert_dists[0].size == 0:
		return state.shape[0] - m_row
	return obj_vert_dists[0][0]

# Returns the number of rows between Mario and the roof (0 if next level is roof)
# if no roof above Mario, return number of rows between Mario and offscreen
# Return None if Mario not on screen
# Always perform None check on return val
def roofVertDistance(state):
	mario_pos = marioPosition(state)
	if mario_pos is None:
		return None
	m_row, m_col = mario_pos

	# get the rows in Mario's column with objects, if any
	col_contents = state[:m_row, m_col]
	obj_vert_dists = np.nonzero(col_contents == 1)

	if obj_vert_dists[0].size == 0:
		return m_row + 1
	return m_row - obj_vert_dists[0][-1]

# Returns the # of columns to the right of Mario for which
# there exists at least one object (ground=1) at a height lower than Mario
def groundRightDistance(state):
	mario_pos = marioPosition(state)
	if mario_pos is None:
		return None
	m_row, m_col = mario_pos

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
def groundLeftDistance(state):
	mario_pos = marioPosition(state)
	if mario_pos is None:
		return None
	m_row, m_col = mario_pos

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
def distLeftEnemy(state):
	mario_pos = marioPosition(state)
	if mario_pos is None:
		return None
	m_row, m_col = mario_pos

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
def distRightEnemy(state):
	mario_pos = marioPosition(state)
	if mario_pos is None:
		return None
	m_row, m_col = mario_pos

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
def distUpEnemy(state):
	mario_pos = marioPosition(state)
	if mario_pos is None:
		return None
	m_row, m_col = mario_pos

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
def distDownEnemy(state):
	mario_pos = marioPosition(state)
	if mario_pos is None:
		return None
	m_row, m_col = mario_pos

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
def groundBelow(state):
	mario_pos = marioPosition(state)
	if mario_pos is None:
		return None
	m_row, m_col = mario_pos

	# get the rows in Mario's column with objects, if any
	col_contents = state[m_row:, m_col]
	ground_below = np.nonzero(col_contents == 1)

	if ground_below[0].size == 0:
		return 0
	return 1

# Return whether Mario can jump in his position (1=true)
def canJump(state):
	mario_pos = marioPosition(state)
	if mario_pos is None:
		return None
	m_row, m_col = mario_pos

	if m_row > 0:
		if state[m_row-1, m_col] == 0:
			return 1
		return 0
	return 1

# Return whether Mario can move right in his position (1=true)
def canMoveRight(state):
	mario_pos = marioPosition(state)
	if mario_pos is None:
		return None
	m_row, m_col = mario_pos

	if m_col < state.shape[1] - 1:
		if state[m_row, m_col + 1] == 0:
			return 1
		return 0
	return 1

# Return whether Mario can move left in his position (1=true)
def canMoveLeft(state):
	mario_pos = marioPosition(state)
	if mario_pos is None:
		return None
	m_row, m_col = mario_pos

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

# TESTING FUNCTIONS

def test_bounds(state, left, right, above, below):
	assert groundLeftDistance(state) == left
	assert groundRightDistance(state) == right
	assert roofVertDistance(state) == above
	assert groundVertDistance(state) == below

def test_enemy_dists(state, dLeft, dRight, dUp, dDown):
	assert distLeftEnemy(state) == dLeft
	assert distRightEnemy(state) == dRight
	assert distUpEnemy(state) == dUp
	assert distDownEnemy(state) == dDown

def main():
	print "Running boundary tests"
	a = np.array([[1,1,1], [3,0,0], [1,1,1]])
	b = np.array([[1,1,1], [0,3,0], [1,1,1]])
	c = np.array([[1,1,1], [0,0,3], [1,1,1]])

	test_bounds(a, 0, 2, 1, 1)
	test_bounds(b, 1, 1, 1, 1)
	test_bounds(c, 2, 0, 1, 1)

	a = np.array([[1,1,1], [3,0,0], [0,0,0]])
	b = np.array([[1,1,1], [0,3,0], [0,0,0]])
	c = np.array([[1,1,1], [0,0,3], [0,0,0]])

	test_bounds(a, 0, 0, 1, 2)
	test_bounds(b, 0, 0, 1, 2)
	test_bounds(c, 0, 0, 1, 2)

	a = np.array([[0,0,0], [3,0,0], [1,1,1]])
	b = np.array([[0,0,0], [0,3,0], [1,1,1]])
	c = np.array([[0,0,0], [0,0,3], [1,1,1]])

	test_bounds(a, 0, 2, 2, 1)
	test_bounds(b, 1, 1, 2, 1)
	test_bounds(c, 2, 0, 2, 1)

	a = np.array([[1,0,1], [3,0,0], [1,0,1]])
	b = np.array([[1,0,1], [0,3,0], [1,0,1]])
	c = np.array([[1,0,1], [0,0,3], [1,0,1]])

	test_bounds(a, 0, 0, 1, 1)
	test_bounds(b, 1, 1, 2, 2)
	test_bounds(c, 0, 0, 1, 1)

	a = np.array([[0,0,0,0,0], [0,0,0,0,0], [0,1,3,1,0], [0,0,0,1,0], [1,1,0,0,0]])
	b = np.array([[0,1,0,0,0], [1,3,1,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,1,1,1,1]])
	c = np.array([[0,0,0,1,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,1,3,1], [0,1,1,0,0]])

	test_bounds(a, 2, 1, 3, 3)
	test_bounds(b, 0, 3, 1, 3)
	test_bounds(c, 2, 0, 3, 2)

	print "Passed all boundary tests!"

	print "Running enemy distance tests"

	a = np.array([[1,1,1], [3,0,0], [1,1,1]])
	b = np.array([[1,1,1], [0,3,0], [1,1,1]])
	c = np.array([[1,1,1], [1,1,1], [0,0,3]])

	test_enemy_dists(a, 1, 3, 2, 2)
	test_enemy_dists(b, 2, 2, 2, 2)
	test_enemy_dists(c, 3, 1, 3, 1)

	a = np.array([[1,1,1], [3,0,0], [0,0,2]])
	b = np.array([[0,2,0], [0,3,0], [1,1,1]])
	c = np.array([[1,1,1], [0,0,3], [2,0,0]])

	test_enemy_dists(a, 1, 2, 2, 1)
	test_enemy_dists(b, 0, 0, 1, 2)
	test_enemy_dists(c, 2, 1, 2, 1)

	a = np.array([[1,2,0], [1,3,0], [1,2,1]])
	b = np.array([[0,0,0], [2,3,0], [2,1,1]])
	c = np.array([[0,0,1], [2,3,2], [1,1,1]])

	test_enemy_dists(a, 0, 0, 1, 1)
	test_enemy_dists(b, 1, 2, 0, 0)
	test_enemy_dists(c, 1, 1, 0, 0)

	print "Passed all enemy distance tests"

	print "All tests passed!"

if __name__ == "__main__": main()
