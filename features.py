import numpy as np

# Returns the vertical distance from Mario to the ground
# State is np array
# mario is 3, ground is first 1 below Mario
# None if no ground directly below

# Returns mario's position as row, col pair
# Returns None if Mario not on map
# Always perform None check on return val
def marioPosition(state):
	rows, cols = np.nonzero(state == 3)
	if rows.size == 0 or cols.size == 0:
		return None
	return rows[0], cols[0]

# Returns vert distance from ground
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
		return None
	return obj_vert_dists[0][0]

# Returns vert distance from ground
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
		return None
	return m_row - obj_vert_dists[0][-1]

