
import gymnasium as gym
import numpy as np
import torch
import copy 
from gymnasium import spaces

class Player():
    def __init__(self, id, token):
        self.id = id
        self.token = token
        

class Token():
    def __init__(self, symbol, number):
        self.number = number
        self.symbol = symbol
        
        
class Connect4Env(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super().__init__()
        self.name = 'connect4'

        self.rows = 6
        self.cols = 7
        self.n_players = 2
        self.grid_shape = (self.rows, self.cols)
        self.num_squares = self.rows * self.cols
        self.action_space = spaces.Discrete(self.cols)
        self.observation_space = spaces.Box(low=0, high=1, shape = (5, self.rows, self.cols), dtype=np.uint8)
        

    @property
    def observation(self):
        if self.current_player.token.number == 1:
            position_1 = np.array([1 if x.number == 1 else 0  for x in self.board]).reshape(self.grid_shape)
            position_2 = np.array([1 if x.number == -1 else 0 for x in self.board]).reshape(self.grid_shape)
            position_3 = np.zeros(self.grid_shape)
        else:
            position_1 = np.array([1 if x.number == -1 else 0 for x in self.board]).reshape(self.grid_shape)
            position_2 = np.array([1 if x.number == 1 else 0 for x in self.board]).reshape(self.grid_shape)
            position_3 = np.ones(self.grid_shape)

        position_4 = self.get_win_cell_plane()
        position_5 = self.get_win_cell_plane(current=False)

        out = np.stack([position_1, position_2, position_3, position_4, position_5], axis = 0).astype(np.uint8)
        return out

    @property
    def legal_actions(self):
        legal_actions = []
        for action_num in range(self.action_space.n):
            legal = self.is_legal(action_num)
            legal_actions.append(legal)
            
        return torch.tensor(legal_actions).to(torch.uint8)

    def get_win_cell_plane(self, current=True):
        plane = np.zeros(self.num_squares)
        for action_num in range(self.action_space.n):
            if self.is_legal(action_num):
                aux_board = copy.deepcopy(self.board)
                square = self.get_square(aux_board, action_num)
                if current:
                    aux_board[square] = self.current_player.token
                    r, done = self.check_game_over(board=aux_board)
                else:
                    opp_player = (self.current_player_num + 1) % 2
                    aux_board[square] = self.players[opp_player].token
                    r, done = self.check_game_over(board=aux_board, player=opp_player)    
                if done and r == 1:
                    plane[square] = 1

        return plane.reshape(self.grid_shape)

    def is_legal(self, action_num):
        if self.board[action_num].number==0:
            return 1
        else:
            return 0

    def square_is_player(self, board, square, player):
        return board[square].number == self.players[player].token.number

    def check_game_over(self, board = None , player = None):

        if board is None:
            board = self.board

        if player is None:
            player = self.current_player_num

        for x,y,z,a in WINNERS:
            if self.square_is_player(board, x, player) and self.square_is_player(board, y, player) and self.square_is_player(board, z, player) and self.square_is_player(board, a, player):
                return 1, True

        if self.turns_taken == self.num_squares:
            return  0, True

        return 0, False #-0.01 here to encourage choosing the win?

    def get_square(self, board, action):
        for height in range(1, self.rows + 1):
            square = self.num_squares - (height * self.cols) + action
            if board[square].number == 0:
                return square

    @property
    def current_player(self):
        return self.players[self.current_player_num]

    def step(self, action):
        
        reward = [0,0]
        
        # check move legality
        board = self.board
        
        if not self.is_legal(action): 
            done = True
            reward = [0,0]
            reward[self.current_player_num] = -1
        else:
            square = self.get_square(board, action)
            board[square] = self.current_player.token

            self.turns_taken += 1
            r, done = self.check_game_over()
            reward = [-r,-r]
            reward[self.current_player_num] = r

        self.done = done

        if not done:
            self.current_player_num = (self.current_player_num + 1) % 2

        return self.observation, reward, done, {}

    def reset(self):
        self.board = [Token('.', 0)] * self.num_squares
        self.players = [Player('1', Token('X', 1)), Player('2', Token('O', -1))]
        self.current_player_num = 0
        self.turns_taken = 0
        self.done = False
        return self.observation
    

    def render(self, mode='human', close=False):
        print('-------------------------')
        for i in range(0,self.num_squares,self.cols):
            print(' '.join([x.symbol for x in self.board[i:(i+self.cols)]]))

WINNERS = [
			[0,1,2,3],
			[1,2,3,4],
			[2,3,4,5],
			[3,4,5,6],
			[7,8,9,10],
			[8,9,10,11],
			[9,10,11,12],
			[10,11,12,13],
			[14,15,16,17],
			[15,16,17,18],
			[16,17,18,19],
			[17,18,19,20],
			[21,22,23,24],
			[22,23,24,25],
			[23,24,25,26],
			[24,25,26,27],
			[28,29,30,31],
			[29,30,31,32],
			[30,31,32,33],
			[31,32,33,34],
			[35,36,37,38],
			[36,37,38,39],
			[37,38,39,40],
			[38,39,40,41],

			[0,7,14,21],
			[7,14,21,28],
			[14,21,28,35],
			[1,8,15,22],
			[8,15,22,29],
			[15,22,29,36],
			[2,9,16,23],
			[9,16,23,30],
			[16,23,30,37],
			[3,10,17,24],
			[10,17,24,31],
			[17,24,31,38],
			[4,11,18,25],
			[11,18,25,32],
			[18,25,32,39],
			[5,12,19,26],
			[12,19,26,33],
			[19,26,33,40],
			[6,13,20,27],
			[13,20,27,34],
			[20,27,34,41],

			[3,9,15,21],
			[4,10,16,22],
			[10,16,22,28],
			[5,11,17,23],
			[11,17,23,29],
			[17,23,29,35],
			[6,12,18,24],
			[12,18,24,30],
			[18,24,30,36],
			[13,19,25,31],
			[19,25,31,37],
			[20,26,32,38],

			[3,11,19,27],
			[2,10,18,26],
			[10,18,26,34],
			[1,9,17,25],
			[9,17,25,33],
			[17,25,33,41],
			[0,8,16,24],
			[8,16,24,32],
			[16,24,32,40],
			[7,15,23,31],
			[15,23,31,39],
			[14,22,30,38],
			]