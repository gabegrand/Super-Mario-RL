from features import *
import numpy as np
import sys

# TESTING

def testMarioPosition():
	a = np.array([[1,1,1], [1,1,0], [0,2,1]])
	b = np.array([[1,2,1], [0,0,3], [1,1,2]])
	assert marioPosition(a) == None
	assert marioPosition(b) == (1, 2)

# Action mapping
MAPPING = {
    0: [0, 0, 0, 0, 0, 0],  # NOOP
    1: [1, 0, 0, 0, 0, 0],  # Up
    2: [0, 0, 1, 0, 0, 0],  # Down
    3: [0, 1, 0, 0, 0, 0],  # Left
    4: [0, 1, 0, 0, 1, 0],  # Left + A
    5: [0, 1, 0, 0, 0, 1],  # Left + B
    6: [0, 1, 0, 0, 1, 1],  # Left + A + B
    7: [0, 0, 0, 1, 0, 0],  # Right
    8: [0, 0, 0, 1, 1, 0],  # Right + A
    9: [0, 0, 0, 1, 0, 1],  # Right + B
    10: [0, 0, 0, 1, 1, 1],  # Right + A + B
    11: [0, 0, 0, 0, 1, 0],  # A
    12: [0, 0, 0, 0, 0, 1],  # B
    13: [0, 0, 0, 0, 1, 1],  # A + B
}

def checkMovement(prev, curr, a_num, up, down, left, right):
	assert movingUp(prev, curr, a_num) == up
	assert movingDown(prev, curr, a_num) == down
	assert movingLeft(prev, curr, a_num) == left
	assert movingRight(prev, curr, a_num) == right

def testMarioDirection():
	#Init
	prev = np.array([[1,1,1], [0,0,0], [1,1,1]])
	curr = np.array([[1,1,1], [0,3,0], [1,1,1]])
	checkMovement(prev, curr, 0, 0, 0, 0, 0)

	# Up
	prev = np.array([[1,1,1], [0,0,0], [1,3,1]])
	curr = np.array([[1,1,1], [0,3,0], [1,0,1]])
	checkMovement(prev, curr, 1, 1, 0, 0, 0)
	checkMovement(prev, curr, 11, 1, 0, 0, 0)

	# TODO case --- should up be 0 or 1 here?
	prev = np.array([[1,0,1], [0,3,0], [1,1,1]])
	curr = np.array([[1,0,1], [0,3,0], [1,1,1]])
	checkMovement(prev, curr, 11, 0, 0, 0, 0)

	prev = np.array([[1,1,1], [0,3,0], [1,1,1]])
	curr = np.array([[1,1,1], [0,3,0], [1,1,1]])
	checkMovement(prev, curr, 11, 0, 0, 0, 0)

	# Up/right
	prev = np.array([[1,1,1], [0,0,0], [1,3,1]])
	curr = np.array([[1,1,1], [0,0,3], [1,0,1]])
	checkMovement(prev, curr, 8, 1, 0, 0, 1)

	# Right
	prev = np.array([[1,1,1], [0,0,0], [1,3,0]])
	curr = np.array([[1,1,1], [0,0,0], [1,0,3]])
	checkMovement(prev, curr, 7, 0, 0, 0, 1)
	checkMovement(prev, curr, 8, 0, 0, 0, 1)
	prev = np.array([[1,0,1], [0,3,0], [1,1,1]])
	curr = np.array([[1,0,1], [0,3,0], [1,1,1]])
	checkMovement(prev, curr, 9, 0, 0, 0, 1)
	prev = np.array([[1,0,1], [1,3,1], [1,1,1]])
	curr = np.array([[1,0,1], [1,3,1], [1,1,1]])
	checkMovement(prev, curr, 10, 0, 0, 0, 0)

	# Down/right
	prev = np.array([[1,1,1], [1,3,1], [1,1,0]])
	curr = np.array([[1,1,1], [1,0,1], [1,1,3]])
	checkMovement(prev, curr, 0, 0, 1, 0, 1)

	# Down
	prev = np.array([[1,1,1], [0,3,0], [1,0,1]])
	curr = np.array([[1,1,1], [0,0,0], [1,3,1]])
	checkMovement(prev, curr, 2, 0, 1, 0, 0)
	checkMovement(prev, curr, 2, 0, 1, 0, 0)

	# TODO case: should down be 0 or 1 here?
	prev = np.array([[1,1,1], [0,3,0], [1,0,1]])
	curr = np.array([[1,1,1], [0,3,0], [1,0,1]])
	checkMovement(prev, curr, 2, 0, 0, 0, 0)

	prev = np.array([[1,1,1], [0,3,0], [1,1,1]])
	curr = np.array([[1,1,1], [0,3,0], [1,1,1]])
	checkMovement(prev, curr, 2, 0, 0, 0, 0)

	# Down/left
	prev = np.array([[1,1,1], [1,3,1], [0,1,1]])
	curr = np.array([[1,1,1], [1,0,1], [3,1,1]])
	checkMovement(prev, curr, 0, 0, 1, 1, 0)

	# Left
	prev = np.array([[1,1,1], [0,0,0], [0,3,1]])
	curr = np.array([[1,1,1], [0,0,0], [3,0,1]])
	checkMovement(prev, curr, 3, 0, 0, 1, 0)
	checkMovement(prev, curr, 4, 0, 0, 1, 0)

	# TODO case: should left be 0 or 1 here?
	prev = np.array([[1,0,1], [0,3,0], [1,1,1]])
	curr = np.array([[1,0,1], [0,3,0], [1,1,1]])
	checkMovement(prev, curr, 5, 0, 0, 0, 0)

	prev = np.array([[1,0,1], [1,3,0], [1,1,1]])
	curr = np.array([[1,0,1], [1,3,0], [1,1,1]])
	checkMovement(prev, curr, 6, 0, 0, 0, 0)

	# Up/Left
	prev = np.array([[1,1,1], [0,0,0], [1,3,1]])
	curr = np.array([[1,1,1], [3,0,0], [1,0,1]])
	checkMovement(prev, curr, 4, 1, 0, 1, 0)

	# Stationary
	prev = np.array([[0,0,0], [0,3,0], [0,0,0]])
	curr = np.array([[0,0,0], [0,3,0], [0,0,0]])
	checkMovement(prev, curr, 0, 0, 0, 0, 0)

def testMarioStatus():
	info = {'time': 3, 'player_status': -1}
	assert marioStatus(info) == -1
	info = {}
	assert marioStatus(info) == None
	info = {'player_status': 2}
	assert marioStatus(info) == 2

def testTimeRemaining():
	info = {'time': -1, 'player_status': 1}
	assert timeRemaining(info) == -1
	info = {'player_status': 1}
	assert timeRemaining(info) == None
	info = {'time': 2}
	assert timeRemaining(info) == 2

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

def testEnemyOnScreen():
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

def testCanMoveLeft():
	a = np.array([[1,1,1], [3,0,0], [1,1,1]])
	b = np.array([[1,1,1], [0,3,0], [1,1,1]])
	c = np.array([[1,1,1], [0,1,3], [1,1,1]])

	assert canMoveLeft(a) == 0
	assert canMoveLeft(b) == 1
	assert canMoveLeft(c) == 0


def testCanMoveRight():
	a = np.array([[1,1,1], [3,1,0], [1,1,1]])
	b = np.array([[1,1,1], [0,3,0], [1,1,1]])
	c = np.array([[1,1,1], [0,1,3], [1,1,1]])

	assert canMoveRight(a) == 0
	assert canMoveRight(b) == 1
	assert canMoveRight(c) == 1

def testGroundBelow():
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

	# enemy dist tests

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

	# individual fucntion tests

	testMarioPosition()
	testMarioDirection()
	testCanMoveLeft()
	testCanMoveRight()
	testEnemyOnScreen()
	testGroundBelow()
	testTimeRemaining()
	testMarioStatus()

	print "All tests passed!"

if __name__ == "__main__": main()
