from features import *
import numpy as np
import sys

def checkAnswerPairs(answer_pairs):
	for guess, ans in answer_pairs:
		assert guess >= 0.0
		assert guess <= 1.0
		assert ans >= 0.0
		assert ans <= 1.0
		assert guess == ans

# TESTING

def testMarioPosition():
	a = np.array([[1,1,1], [1,1,0], [0,2,1]])
	b = np.array([[1,2,1], [0,0,3], [1,1,2]])

	ans_a = marioPosition(a)
	ans_b = marioPosition(b)

	assert ans_a == None
	assert ans_b == (1, 2)

def checkMovement(prev, curr, a_num, up, down, left, right):

	u = movingUp(prev, curr, a_num)
	d = movingDown(prev, curr, a_num)
	l = movingLeft(prev, curr, a_num)
	r = movingRight(prev, curr, a_num)

	checkAnswerPairs([(u, up), (d, down), (l, left), (r, right)])

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

def test_bounds(state, left, right, above, below):
	l = groundLeftDistance(state)
	left = float(left) / (state.shape[1] - 1)

	r = groundRightDistance(state)
	right = float(right) / (state.shape[1] - 1)

	a = roofVertDistance(state)
	above = float(above) / state.shape[0]

	b = groundVertDistance(state)
	below = float(below) / state.shape[0]

	checkAnswerPairs([(a, above), (b, below), (l, left), (r, right)])

def test_enemy_dists(state, dLeft, dRight, dUp, dDown):
	l = distLeftEnemy(state)
	dLeft = float(dLeft) / state.shape[1]

	r = distRightEnemy(state)
	dRight = float(dRight) / state.shape[1]


	u = distUpEnemy(state)
	dUp = float(dUp) / state.shape[0]


	d = distDownEnemy(state)
	dDown = float(dDown) / state.shape[0]

	checkAnswerPairs([(l, dLeft), (r, dRight), (u, dUp), (d, dDown)])

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

	test_bounds(a, 0, 2, 0, 0)
	test_bounds(b, 1, 1, 0, 0)
	test_bounds(c, 2, 0, 0, 0)

	a = np.array([[1,1,1], [3,0,0], [0,0,0]])
	b = np.array([[1,1,1], [0,3,0], [0,0,0]])
	c = np.array([[1,1,1], [0,0,3], [0,0,0]])

	test_bounds(a, 0, 0, 0, 1)
	test_bounds(b, 0, 0, 0, 1)
	test_bounds(c, 0, 0, 0, 1)

	a = np.array([[0,0,0], [3,0,0], [1,1,1]])
	b = np.array([[0,0,0], [0,3,0], [1,1,1]])
	c = np.array([[0,0,0], [0,0,3], [1,1,1]])

	test_bounds(a, 0, 2, 1, 0)
	test_bounds(b, 1, 1, 1, 0)
	test_bounds(c, 2, 0, 1, 0)

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
	test_bounds(b, 0, 3, 0, 2)
	test_bounds(c, 2, 0, 2, 1)

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

	print "All tests passed!"

if __name__ == "__main__": main()
