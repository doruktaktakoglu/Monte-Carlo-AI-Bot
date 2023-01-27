# Student agent: Add your own agent here
from ftplib import parse150
from sqlite3 import Date
from turtle import right
from agents.agent import Agent
from agents.random_agent import RandomAgent
from store import register_agent
import numpy as np
import time
import sys


@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
        self.autoplay = True

    def step(self, chess_board, my_pos, adv_pos, max_step):
        time_start = time.time()
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (x_max, y_max, 4)
        - my_pos: a tuple of (x, y)
        - adv_pos: a tuple of (x, y)
        - max_step: an integer

        You should return a tuple of ((x, y), dir),
        where (x, y) is the next position of your agent and dir is the direction of the wall
        you want to put on.

        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """
        board_size = len(chess_board[0])
        # generating the moves
        possible_moves = []
        current_x = my_pos[0]
        current_y = my_pos[1]
        rightmost = current_x + max_step
        leftmost = current_x - max_step
        upmost = current_y - max_step
        downmost = current_y + max_step
        if(leftmost < 0):
            leftmost = 0
        if (upmost < 0):
            upmost = 0
        if (downmost > board_size - 1):
            downmost = board_size
        if (rightmost > board_size - 1):
            rightmost = board_size
        

        a = leftmost
        while (a <= rightmost):
            b = upmost
            while (b <= downmost):
                end_pos = (a, b)
                for barrier_vector in range(4):
                    # code for checking valid move
                    try:
                        r = end_pos[0]
                        c = end_pos[1]
                        if chess_board[r, c, barrier_vector]:
                            continue
                        if np.array_equal(my_pos, end_pos):
                            possible_moves.append((end_pos, barrier_vector))
                            continue

                        state_queue = [(my_pos, 0)]
                        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
                        visited = {tuple(my_pos)}
                        is_reached = False
                        while state_queue and not is_reached:
                            cur_pos, cur_step = state_queue.pop(0)
                            r = cur_pos[0]
                            c = cur_pos[1]
                            if cur_step == max_step:
                                break
                            for dir, move in enumerate(moves):
                                if chess_board[r, c, dir]:
                                    continue

                                next_pos = (cur_pos[0] + move[0], cur_pos[1] + move[1])
                                if np.array_equal(next_pos, adv_pos) or tuple(next_pos) in visited:
                                    continue
                                if np.array_equal(next_pos, end_pos):
                                    is_reached = True
                                    break

                                visited.add(tuple(next_pos))
                                state_queue.append((next_pos, cur_step + 1))
                        if is_reached:
                            possible_moves.append((end_pos, barrier_vector))
                    except:
                        continue
                    # code for checking valid move
                b += 1
            a += 1
        time_left = 1.5 - (time.time() - time_start)
        # generating the moves
        # dummy return
        # pick one item at random from moves
        my_move = self.get_best_move(possible_moves, chess_board, adv_pos, board_size, time_left)
        return my_move

    def get_best_move(self, possible_moves, chess_board, adv_pos, board_size, max_time):
        # for each move, calculate the score
        # return the move with the highest score
        best_move = possible_moves[0]
        best_score = 0
        time_start = time.time()
        max_time_per_simulation = max(max_time / len(possible_moves), 0.01)
        for move in possible_moves:
            if time.time() - time_start > max_time:
                break
            my_pos = move[0]
            barrier_vector = move[1]
            new_board = chess_board.copy()
            new_board[my_pos[0], my_pos[1], barrier_vector] = True
            total_score = 0
            runs = 0
            for i in range(20):
                if time.time() - time_start > max_time:
                    break
                runs = runs + 1
                p1score, p0score = self.simulate(new_board, my_pos, adv_pos, board_size, max_time_per_simulation)
                if p1score > p0score:
                    total_score += 1
                if p1score < p0score:
                    total_score -= 1
                else:
                    total_score += 0.5
            p0_score = total_score / runs
            if p0_score > best_score:
                best_score = p0_score
                best_move = move

        return best_move

    def simulate(self, board, p0_pos, p1_pos, max_step, max_time):
        chess_board = board.copy()
        # check while the game is not over
        board_size = len(chess_board[0])
        is_end, p0_score, p1_score = self.check_endgame(board_size, p0_pos, p1_pos, chess_board)
        time_start = time.time()
        i = 0
        while (not is_end) and (time.time() - time_start < max_time):
            moves = self.getPossibleMoves(chess_board, p1_pos, p0_pos, max_step)
            # get random move
            if len(moves) == 0:
                break
            random_move = moves[np.random.randint(len(moves))]
            # apply random move to chess_board
            chess_board[random_move[0][0], random_move[0][1], random_move[1]] = True
            # check while the game is not over
            if i % 2 == 0:
                is_end, p0_score, p1_score = self.check_endgame(board_size, p0_pos, p1_pos, chess_board)
            else:
                is_end, p0_score, p1_score = self.check_endgame(board_size, p1_pos, p0_pos, chess_board)
            p0_pos = p1_pos
            p1_pos = random_move[0]
            i = i + 1

        return p0_score, p1_score

    def check_endgame(self,board_size, p0_pos, p1_pos, chess_board):
        """
        Check if the game ends and compute the current score of the agents.

        Returns
        -------
        is_endgame : bool
            Whether the game ends.
        player_1_score : int
            The score of player 1.
        player_2_score : int
            The score of player 2.
        """
        # Union-Find
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        father = dict()
        for r in range(board_size):
            for c in range(board_size):
                father[(r, c)] = (r, c)

        def find(pos):
            if father[pos] != pos:
                father[pos] = find(father[pos])
            return father[pos]

        def union(pos1, pos2):
            father[pos1] = pos2

        for r in range(board_size):
            for c in range(board_size):
                for dir, move in enumerate(
                    moves[1:3]
                ):  # Only check down and right
                    if chess_board[r, c, dir + 1]:
                        continue
                    pos_a = find((r, c))
                    pos_b = find((r + move[0], c + move[1]))
                    if pos_a != pos_b:
                        union(pos_a, pos_b)

        for r in range(board_size):
            for c in range(board_size):
                find((r, c))
        p0_r = find(tuple(p0_pos))
        p1_r = find(tuple(p1_pos))
        p0_score = list(father.values()).count(p0_r)
        p1_score = list(father.values()).count(p1_r)
        if p0_r == p1_r:
            return False, p0_score, p1_score
        return True, p0_score, p1_score


    def getPossibleMoves(self, chess_board, my_pos, adv_pos, max_step):
        board_size = len(chess_board[0])
        possible_moves = []
        current_x = my_pos[0]
        current_y = my_pos[1]
        rightmost = current_x + max_step
        leftmost = current_x - max_step
        upmost = current_y - max_step
        downmost = current_y + max_step
        if(leftmost < 0):
            leftmost = 0
        if (upmost < 0):
            upmost = 0
        if (downmost > board_size - 1):
            downmost = board_size
        if (rightmost > board_size - 1):
            rightmost = board_size
        

        a = leftmost
        while (a <= rightmost):
            b = upmost
            while (b <= downmost):
                end_pos = (a, b)
                for barrier_vector in range(4):
                    # code for checking valid move
                    try:
                        r = end_pos[0]
                        c = end_pos[1]
                        if chess_board[r, c, barrier_vector]:
                            continue
                        if np.array_equal(my_pos, end_pos):
                            possible_moves.append((end_pos, barrier_vector))
                            continue

                        state_queue = [(my_pos, 0)]
                        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
                        visited = {tuple(my_pos)}
                        is_reached = False
                        while state_queue and not is_reached:
                            cur_pos, cur_step = state_queue.pop(0)
                            r = cur_pos[0]
                            c = cur_pos[1]
                            if cur_step == max_step:
                                break
                            for dir, move in enumerate(moves):
                                if chess_board[r, c, dir]:
                                    continue

                                next_pos = (cur_pos[0] + move[0], cur_pos[1] + move[1])
                                if np.array_equal(next_pos, adv_pos) or tuple(next_pos) in visited:
                                    continue
                                if np.array_equal(next_pos, end_pos):
                                    is_reached = True
                                    break

                                visited.add(tuple(next_pos))
                                state_queue.append((next_pos, cur_step + 1))
                        if is_reached:
                            possible_moves.append((end_pos, barrier_vector))
                    except:
                        continue
                    # code for checking valid move
                b += 1
            a += 1

        return possible_moves