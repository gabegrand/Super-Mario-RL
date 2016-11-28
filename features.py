import numpy as np

# Returns mario's position as row, col pair
# Returns None if Mario not on map
# Always perform None check on return val
def marioPosition(state):
	rows, cols = np.nonzero(state == 3)
	if rows.size == 0 or cols.size == 0:
		print "WARNING: Mario is off the map"
		return None
	return rows[0], cols[0]

# Returns the number of rows from Mario to the roof
# Return None if no ground is vertically below Mario
# Always perform None check on return val
def distanceFromGround(state):
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

# Returns the number of rows from Mario to the roof
# Return None if no roof is vertically above Mario
# Always perform None check on return val
def distanceFromRoof(state):
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
def rightGroundDistance(state):
	mario_pos = marioPosition(state)
	if mario_pos is None:
		return None
	m_row, m_col = mario_pos

	dist = 0

	if m_row < state.shape[0]:
		for col in xrange(m_col, state.shape[1]):
			col_contents = state[m_row + 1:, col]
			obj_vert_dists = np.nonzero(col_contents == 1)

			if obj_vert_dists[0].size == 0:
				return dist
			dist += 1
	return dist

# Returns the # of columns to the left of Mario for which
# there exists at least one object (ground=1) at a height lower than Mario
def leftGroundDistance(state):
	mario_pos = marioPosition(state)
	if mario_pos is None:
		return None
	m_row, m_col = mario_pos

	dist = 0

	if m_row < state.shape[0]:
		for col in xrange(0, m_col):
			col_contents = state[m_row + 1:, col]
			obj_vert_dists = np.nonzero(col_contents == 1)

			if obj_vert_dists[0].size == 0:
				return dist
			dist += 1
	return dist