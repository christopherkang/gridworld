import os
import pickle as p

import cv2
import numpy as np

from Agents import linear, random_choice
from Gridworld import Gridworld

ACTION_BANK = ["UP", "LEFT", "DOWN", "RIGHT"]
ACTION_DEF = [[0, -1],
              [-1, 0],
              [0, 1],
              [1, 0]]
ACTION_INFO = [ACTION_BANK, ACTION_DEF]

params = [[[0.5, True, 3, 4],
           [0.5, True, 4, 5],
           [0.5, True, 5, 6],
           [0.7, True, 3, 7],
           [0.5, True, 3, 8],
           [0.3, True, 3, 9],
           [0.5, True, 3, 2], ],
          0]

WORLD_SIZE = (11, 11)

RUN_NUM = input("What run is this? ")


def randomly_create_objects(number_of_objects, reward, xyDimension,
                            reward_map=""):
    xSize = xyDimension[0]
    ySize = xyDimension[1]

    output = []

    if reward_map:
        for i in range(number_of_objects):
            toAdd = [reward_map[i], True, np.random.choice(
                xSize), np.random.choice(ySize)]
            output.append(toAdd)
    else:
        for _ in range(number_of_objects):
            toAdd = [reward, True, np.random.choice(
                xSize), np.random.choice(ySize)]
            output.append(toAdd)
    return output


auto_params = [randomly_create_objects(10, 1, WORLD_SIZE), 0]


def showPotentialAction(environment):
    cv2.imshow('image', cv2.resize(environment.getRepresentation(
        showAgent=True, scaleEnvironment=True), (200, 200),
        interpolation=cv2.INTER_NEAREST))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    xDelta = int(input("xDelta?"))
    yDelta = int(input("yDelta?"))
    cv2.imshow('image', cv2.resize(environment.simulateAction(
        xDelta, yDelta), (200, 200), interpolation=cv2.INTER_NEAREST))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_epoch(agent, world, round_num, epochs, save=True, animate=True):
    GAMMA = 0.9
    MC_matrix = np.zeros((epochs, 4))
    for epoch in range(epochs):
        action_choice = agent.execute_policy()
        reward = world.move_agent(action_choice)
        agent.consume_reward(reward, round_num)
        MC_matrix[epoch, 0] = reward
        MC_matrix[epoch, 1], MC_matrix[epoch,
                                       [2, 3]] = agent.predict_value(return_sums=True)
        # agent.gradient_descent(reward) # use this line for per epoch update
        if animate:
            cv2.imshow('image', cv2.resize(world.get_representation(
                showAgent=True, scaleEnvironment=True), (200, 200),
                interpolation=cv2.INTER_NEAREST))
            cv2.waitKey(20)
        if save:
            agent.save_model(RUN_NUM, round_num, epoch)
    update_matrix = MC_matrix.copy()
    for index in range(epochs):
        run_sum = 0
        for sum_range in range(index, epochs):
            run_sum += GAMMA**(sum_range - index) * MC_matrix[sum_range, 0]
        update_matrix[index, 0] = run_sum
    agent.mc_update(update_matrix)


def run_trial(agent, rounds, epochs, save=True):
    save_path = f"/Models/tmp{RUN_NUM}/"
    if save:
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        p.dump(params, open(f"/Models/tmp{RUN_NUM}/env.txt", "wb"))

    for round_num in range(rounds):
        world = Gridworld(WORLD_SIZE, ACTION_INFO, auto_params)
        agent.set_world(world)
        world.place_agent(10, 10)
        run_epoch(agent, world, round_num, epochs, save=save)
        if save:
            agent.end_round(RUN_NUM, round_num)


def test_agent(directory):
    agent = linear.Linear(
        0.1, 0.9, ACTION_INFO, WORLD_SIZE,
        weight_scheme="ZERO", learning_rate=0.01
    )
    agent.load_model(directory)
    return agent


if __name__ == "__main__":
    agent = linear.Linear(
        0.1, 0.9, ACTION_INFO, WORLD_SIZE,
        weight_scheme="ZERO", learning_rate=0.0005
    )
    # agent = test_agent("/Models/tmp9/r50/modele99.txt")
    # world.load_world("/Models/tmp9/env.txt")
    run_trial(agent, 100, 100, save=True)
