from features import *
import numpy as np
import sys

# TESTING

def _testMarioPosition():
	a = np.array([[1,1,1], [1,1,0], [0,2,1]])
	b = np.array([[1,2,1], [0,0,3], [1,1,2]])
	assert marioPosition(a) == None
	assert marioPosition(b) == (1, 2)

def _testMarioDirection():
	#Init
	prev = None
	curr = (1,1)
	assert marioDirection(prev, curr) == 0
	# Up
	prev = (1,1)
	curr = (0,1)
	assert marioDirection(prev, curr) == 1
	# Up/right
	prev = (1,0)
	curr = (0,1)
	assert marioDirection(prev, curr) == 2
	# Right
	prev = (0,0)
	curr = (0,1)
	assert marioDirection(prev, curr) == 3
	# Down/right
	prev = (0,0)
	curr = (1,1)
	assert marioDirection(prev, curr) == 4
	# Down
	prev = (0,0)
	curr = (1,0)
	assert marioDirection(prev, curr) == 5
	# Down/left
	prev = (0,1)
	curr = (1,0)
	assert marioDirection(prev, curr) == 6
	# Left
	prev = (1,1)
	curr = (1,0)
	assert marioDirection(prev, curr) == 7
	# Up/Left
	prev = (1,1)
	curr = (0,0)
	assert marioDirection(prev, curr) == 8
	# Stationary
	prev = (1,1)
	curr = (1,1)
	assert marioDirection(prev, curr) == 0

def _testMarioStatus():
	info = {'time': 3, 'player_status': -1}
	assert marioStatus(info) == -1
	info = {}
	assert marioStatus(info) == None
	info = {'player_status': 2}
	assert marioStatus(info) == 2

def _testTimeRemaining():
	info = {'time': -1, 'player_status': 1}
	assert timeRemaining(info) == -1
	info = {'player_status': 1}
	assert timeRemaining(info) == None
	info = {'time': 2}
	assert timeRemaining(info) == 2

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

# TODO can we move left if at left edge of screen?
def _testCanMoveLeft():
	a = np.array([[1,1,1], [3,0,0], [1,1,1]])
	b = np.array([[1,1,1], [0,3,0], [1,1,1]])
	c = np.array([[1,1,1], [0,1,3], [1,1,1]])

	assert canMoveLeft(a) == 0
	assert canMoveLeft(b) == 1
	assert canMoveLeft(c) == 0


def _testCanMoveRight():
	a = np.array([[1,1,1], [3,1,0], [1,1,1]])
	b = np.array([[1,1,1], [0,3,0], [1,1,1]])
	c = np.array([[1,1,1], [0,1,3], [1,1,1]])

	assert canMoveRight(a) == 0
	assert canMoveRight(b) == 1
	assert canMoveRight(c) == 1

def _testCanJump():
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

def _testGroundBelow():
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

def main():
	print "Testing feature functions..."

	# state boundary tests

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

	# enemy dist tests

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

	# individual fucntion tests

	_testMarioPosition()
	_testMarioDirection()
	_testCanJump()
	_testCanMoveLeft()
	_testCanMoveRight()
	_testEnemyOnScreen()
	_testGroundBelow()
	_testTimeRemaining()
	_testMarioStatus()

	print "All tests passed!"

if __name__ == "__main__": main()
