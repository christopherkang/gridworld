import numpy as np
import pickle as p


class Gridworld:
    """Gridworld class

    Returns:
        [type] -- [description]
    """

    def __init__(self, worldSize, action_specs, parameters):
        """Create Gridworld.

        The indices go from 0..xSize - 1 and 0..ySize - 1

        Arguments:
            xSize {int} -- size of matrix in axis 1
            ySize {int} -- size of matrix in axis 0
            objects {matrix; n x 3} -- objects to be placed on grid
        """
        self.x_size = worldSize[0]
        self.y_size = worldSize[1]
        self.representation = np.zeros(worldSize)
        self.blocks = np.zeros(worldSize)
        self.collision_penalty = parameters[1]
        self.item_list = np.array(parameters[0])
        self.epoch = 0
        self.ACTION_BANK = action_specs[0]
        self.ACTION_EFFECTS = action_specs[1]
        # reversed because its rows x columns
        self.does_agent_exist = False
        for [value, canEnter, x_coord, y_coord] in parameters[0]:
            self.representation[y_coord, x_coord] = value
            if (not canEnter):
                self.blocks[y_coord, x_coord] = 1

    def __str__(self):
        return self.get_representation(True, True)

    def place_agent(self, xCoord, yCoord):
        """Place agent on map; assumes agent doesn't exist.

        Arguments:
            xCoord {int} -- coordinate of x position
            yCoord {int} -- coordinate of y position

        Returns:
            bool -- returns False if agent exist

        """
        if (self.does_agent_exist):
            return False
        else:
            self.xAgent = xCoord
            self.yAgent = yCoord
            self.does_agent_exist = True
            self.create_proximity_map()

    def distance_to_objects(self, xCoord, yCoord):
        """Return matrix with distance to relevant objects

        Arguments:
            xCoord {int} -- xCoord to compare object to
            yCoord {int} -- yCoord to compare object to

        Returns:
            matrix -- matrix of each item and it's value / x delta / y delta
        """

        distance_matrix = [[row[0], row[1] - xCoord, row[2] - yCoord]
                           for row in self.item_list if row[1]]
        return distance_matrix

    def get_agent_coords(self):
        """Return agent coordinates (assumes agent has been placed).

        Returns:
            (int, int) -- coordinates of agent

        """
        return (self.xAgent, self.yAgent)

    def move_possible(self, xyTuple):
        """Identify if a move is possible. Requires agent to be initialized

        Arguments:
            xyTuple {Tuple} -- change in x and y position

        Returns:
            Bool -- returns whether the move is possible
        """

        xEnd = self.xAgent + xyTuple[0]
        yEnd = self.yAgent + xyTuple[1]
        if (xEnd < 0 or xEnd > self.x_size - 1 or
            yEnd < 0 or yEnd > self.y_size - 1 or
                self.blocks[yEnd, xEnd] == 1):
            return False
        return True

    def move_agent(self, action):
        """Move agent given string action.

        If the agent would go off the grid, the agent maintains its current
            position

        Arguments:
            action {int} -- argument for how to move. refers to an index in a
                common action index bank

        Returns:
            int -- Returns reward after action

        """
        if (self.does_agent_exist):
            self.update_proximity_map(self.ACTION_EFFECTS[action])
            return self.appropriate_move(self.ACTION_EFFECTS[action])
        else:
            raise Exception("Agent does not exist!")

    def appropriate_move(self, xyTuple, debugging=False):
        """Decide if a move is appropriate and take it if necessary

        Arguments:
            xyTuple {Tuple} -- change in x and y position

        Returns:
            int -- return reward of moving to that square
        """

        xEnd = self.xAgent + xyTuple[0]
        yEnd = self.yAgent + xyTuple[1]
        if debugging:
            print(f"Target end position: ({xEnd}, {yEnd})")
        if (self.move_possible(xyTuple)):
            self.xAgent = xEnd
            self.yAgent = yEnd
            self.epoch += 1
            output = self.representation[self.yAgent, self.xAgent]

            # clear item
            self.representation[self.yAgent, self.xAgent] = 0

            # remove item from proximity map
            for row in self.prox_map:
                if (row[1] == self.xAgent and row[2] == self.yAgent):
                    row[0] = 0
            return output
        return self.collision_penalty

    def get_representation(self, showAgent=False, scaleEnvironment=False):
        """Return matrix form of total environment.

        Keyword Arguments:
            scaleEnvironment {bool} -- whether to represent the rewards in a
                scaled way to allow better visual representation
            showAgent {bool} -- whether to show the agent in the
                matrix representation (default: {False})

        Returns:
            matrix -- matrix representing the environment

        """
        output = np.array(self.representation)
        if (scaleEnvironment):
            output /= max(output.max(), 1) * 2
        if (showAgent):
            output[self.yAgent, self.xAgent] = 255
        return output

    def return_vision(self, xDist, yDist):
        """Return what an agent "would" be able to see

        Arguments:
            xDist {int} -- distance the agent can see in the x direction
            yDist {int} -- distance the agent can see in the y direction

        Returns:
            matrix or boolean -- matrix of the vision with the reward values,
            false if there is an unexpected error
        """

        if (self.does_agent_exist):
            output = np.zeros((2 * yDist + 1, 2 * xDist + 1))

            for xCoord in range(-xDist, xDist + 1):
                for yCoord in range(-yDist, yDist + 1):
                    xPos = self.xAgent + xCoord
                    yPos = self.yAgent + yCoord
                    if (xPos < 0 or xPos >= self.x_size) or (
                            yPos < 0 or yPos >= self.y_size):
                        output[yCoord + yDist, xCoord + xDist] = -np.infty
                    else:
                        output[yCoord + yDist, xCoord +
                               xDist] = self.representation[yPos, xPos]
            return output
        else:
            return False

    def simulate_action(self, xyTuple):
        """Return simulated matrix if a specific move was made

        Arguments:
            xyTuple {Tuple} -- change in x and y position

        Returns:
            matrix -- matrix representation of environment
        """

        output = np.array(self.get_representation(True, True))
        if (self.move_possible(xyTuple)):
            output[self.yAgent, self.xAgent] = 0
            output[self.yAgent + xyTuple[1], self.xAgent + xyTuple[0]] = 1
            return output
        else:
            return output

    def get_epoch(self):
        """Return the epoch of the current trial

        Returns:
            int -- epoch of the world
        """

        return self.epoch

    def create_proximity_map(self):
        """Create a proximity map showing the distance to each of the objects
        and their values
        """

        self.prox_map = np.array([row[[0, 2, 3]]
                                  for row in self.item_list if row[1]])
        self.prox_map[:, 1] -= (self.xAgent)
        self.prox_map[:, 2] -= (self.yAgent)

    def update_proximity_map(self, xyTuple, speculative=False):
        """Update proximity map

        Arguments:
            xyTuple {Tuple} -- change in x and y position

        Keyword Arguments:
            speculative {bool} -- Choose whether to update the proximity map
                or speculate the action and output the would-be result
                (default: {False})

        Returns:
            matrix -- matrix of the map (only if speculative True)
        """

        if speculative:
            return self.calculate_prox_map()
        else:
            self.prox_map = self.calculate_prox_map()

    def calculate_prox_map(self):
        prox_map = np.array([row[[0, 2, 3]]
                             for row in self.item_list if row[1]])
        prox_map[:, 1] -= (self.xAgent)
        prox_map[:, 2] -= (self.yAgent)
        return prox_map

    def load_world(self, directory):
        parameters = p.load(open(directory, 'rb'))
        self.item_list = parameters[0]
        self.collision_penalty = parameters[1]
