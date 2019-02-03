import copy
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


def run_trial(agent, world_template, rounds, epochs, save=True):
    world = copy.deepcopy(world_template)
    save_path = f"/Models/tmp{RUN_NUM}/"
    if save:
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        p.dump(params, open(f"/Models/tmp{RUN_NUM}/env.txt", "wb"))

    for roundNum in range(rounds):
        agent.set_world(world)
        world.place_agent(10, 10)
        for epoch in range(epochs):
            agent.consume_reward(world.move_agent(
                agent.execute_policy()))
            cv2.imshow('image', cv2.resize(world.get_representation(
                showAgent=True, scaleEnvironment=True), (200, 200),
                interpolation=cv2.INTER_NEAREST))
            cv2.waitKey(20)
            if save:
                agent.save_model(RUN_NUM, roundNum, epoch)
        if save:
            agent.end_round(RUN_NUM, roundNum)


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
        weight_scheme="ZERO", learning_rate=0.01
    )
    # agent = test_agent("/Models/tmp9/r50/modele99.txt")
    world = Gridworld(WORLD_SIZE, ACTION_INFO, auto_params)
    # world.load_world("/Models/tmp9/env.txt")
    run_trial(agent, world, 100, 100, save=True)
