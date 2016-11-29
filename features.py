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

# Returns the number of rows between Mario and the ground (0 if next level is ground)
# if no ground below Mario, return number of rows between Mario and offscreen
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
		return state.shape[0] - m_row - 1
	return obj_vert_dists[0][0] - 1

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
		return m_row
	return m_row - obj_vert_dists[0][-1] - 1

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

# horizontal distance to nearest enemy (right is positive)
def horDistEnemy(state):
	mario_pos = marioPosition(state)
	if mario_pos is None:
		return None
	m_row, m_col = mario_pos

	# if no enemies, return right horizontal distance remaining on screen
	e_rows, e_cols = np.nonzero(state == 2)
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

	# look to the left
	for col_num in xrange(m_col - 1, -1, -1):
		# if we're already farther away than nearest, break
		if abs(m_col - col_num) > nearest:
			break

		col_contents = state[:, col_num]
		enemies = np.nonzero(col_contents == 2)

		# if we've found an enemy, we're as close as we'll get
		if enemies[0].size > 0:
			if abs(m_col - col_num) < nearest:
				nearest = col_num - m_col
			break

	return nearest

# horizontal distance to nearest enemy (down is positive b/c of indexing)
def vertDistEnemy(state):
	mario_pos = marioPosition(state)
	if mario_pos is None:
		return None
	m_row, m_col = mario_pos

	# if no enemies, return vertical distance to bottom of screen
	e_rows, e_cols = np.nonzero(state == 2)
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

	# look above
	for row_num in xrange(m_row - 1, -1, -1):
		# if we're already farther away than nearest, break
		if abs(m_row - row_num) > nearest:
			break

		row_contents = state[row_num, :]
		enemies = np.nonzero(row_contents == 2)

		# if we've found an enemy, we're as close as we'll get
		if enemies[0].size > 0:
			if abs(m_row - row_num) < nearest:
				nearest = row_num - m_row
			break

	return nearest

# Return the number of enemies on the screen
def numEnemiesOnScreen(state):
	rows, cols = np.nonzero(state == 2)
	if rows.size == 0 or cols.size == 0:
		return 0
	return len(rows)

# Return the status of mario
# 0 = small, 1 = big, 2+ = fireball
def marioStatus(info):
	return _fetchEntry(info, "player_status")

# Return the number of seconds remaining in the level
def timeRemaining(info):
	return _fetchEntry(info, "time")

# Return the horizontal distance moved from the start
def distanceFromStart(info):
	return _fetchEntry(info, "distance")

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

def test_bounds(state, left, right, below, above):
	assert groundLeftDistance(state) == left
	assert groundRightDistance(state) == right
	assert groundVertDistance(state) == below
	assert roofVertDistance(state) == above

def test_enemy_dists(state, hDistEnemy, vDistEnemy):
	assert horDistEnemy(state) == hDistEnemy
	assert vertDistEnemy(state) == vDistEnemy

def main():
	print "Running boundary tests"
	a = np.array([[1,1,1], [3,0,0], [1,1,1]])
	b = np.array([[1,1,1], [0,3,0], [1,1,1]])
	c = np.array([[1,1,1], [0,0,3], [1,1,1]])

	test_bounds(a, 0, 2, 0, 0)
	test_bounds(b, 1, 1, 0, 0)
	test_bounds(c, 2, 0, 0, 0)

	a = np.array([[1,1,1], [3,0,0], [0,0,0]])
	b = np.array([[1,1,1], [0,3,0], [0,0,0]])
	c = np.array([[1,1,1], [0,0,3], [0,0,0]])

	test_bounds(a, 0, 0, 1, 0)
	test_bounds(b, 0, 0, 1, 0)
	test_bounds(c, 0, 0, 1, 0)

	a = np.array([[0,0,0], [3,0,0], [1,1,1]])
	b = np.array([[0,0,0], [0,3,0], [1,1,1]])
	c = np.array([[0,0,0], [0,0,3], [1,1,1]])

	test_bounds(a, 0, 2, 0, 1)
	test_bounds(b, 1, 1, 0, 1)
	test_bounds(c, 2, 0, 0, 1)

	a = np.array([[1,0,1], [3,0,0], [1,0,1]])
	b = np.array([[1,0,1], [0,3,0], [1,0,1]])
	c = np.array([[1,0,1], [0,0,3], [1,0,1]])

	test_bounds(a, 0, 0, 0, 0)
	test_bounds(b, 1, 1, 1, 1)
	test_bounds(c, 0, 0, 0, 0)

	a = np.array([[0,0,0,0,0], [0,0,0,0,0], [0,1,3,1,0], [0,0,0,1,0], [1,1,0,0,0]])
	b = np.array([[0,1,0,0,0], [1,3,1,0,0], [0,0,0,0,0], [0,0,0,0,0], [0,1,1,1,1]])
	c = np.array([[0,0,0,1,0], [0,0,0,0,0], [0,0,0,0,0], [0,0,1,3,1], [0,1,1,0,0]])

	test_bounds(a, 2, 1, 2, 2)
	test_bounds(b, 0, 3, 2, 0)
	test_bounds(c, 2, 0, 1, 2)

	print "Passed all boundary tests!"

	print "Running enemy distance tests"

	a = np.array([[1,1,1], [3,0,0], [1,1,1]])
	b = np.array([[1,1,1], [0,3,0], [1,1,1]])
	c = np.array([[1,1,1], [1,1,1], [0,0,3]])

	test_enemy_dists(a, 3, 2)
	test_enemy_dists(b, 2, 2)
	test_enemy_dists(c, 1, 1)

	a = np.array([[1,1,1], [3,0,0], [0,0,2]])
	b = np.array([[0,2,0], [0,3,0], [1,1,1]])
	c = np.array([[1,1,1], [0,0,3], [2,0,0]])

	test_enemy_dists(a, 2, 1)
	test_enemy_dists(b, 0, -1)
	test_enemy_dists(c, -2, 1)

	a = np.array([[1,2,0], [1,3,0], [1,2,1]])
	b = np.array([[0,0,0], [2,3,0], [2,1,1]])
	c = np.array([[0,0,1], [2,3,2], [1,1,1]])

	test_enemy_dists(a, 0, 1)
	test_enemy_dists(b, -1, 0)
	test_enemy_dists(c, 1, 0)

	print "Passed all enemy distance tests"

	print "All tests passed!"

if __name__ == "__main__": main()
