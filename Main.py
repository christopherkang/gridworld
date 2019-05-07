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

WORLD_SIZE = (5, 5)

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


# auto_params = [randomly_create_objects(
#     2, 1, WORLD_SIZE) + randomly_create_objects(2, -1, WORLD_SIZE), 0]

# auto_params = [randomly_create_objects(
#     3, 1, WORLD_SIZE), 0]

auto_params = [[[1, True, 3, 2], [-1, True, 1, 1], [-1, True, 2, 0]], 0]


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


def run_epoch(agent, world, round_num, epochs,
              directory, save=True, animate=True):
    GAMMA = 0.9
    MC_matrix = np.zeros((epochs, 4))

    master_partials = []

    for epoch in range(epochs):
        action_choice = agent.execute_policy()
        reward = world.move_agent(action_choice)
        agent.consume_reward(reward, round_num)

        MC_matrix[epoch, 0] = reward

        prediction, weight_partials = agent.predict_value(return_sums=True)

        master_partials.append(weight_partials)

        MC_matrix[epoch, 1] = prediction
        # MC_matrix[epoch, 2] = dist_matrix[0]
        # MC_matrix[epoch, 3] = dist_matrix[1]

        # MC_matrix[epoch, 1], MC_matrix[epoch, [2, 3]] =
        # agent.gradient_descent(reward) # use this line for per epoch update

        if animate:
            cv2.imshow('image', cv2.resize(world.get_representation(
                showAgent=True, scaleEnvironment=True), (200, 200),
                interpolation=cv2.INTER_NEAREST))
            cv2.waitKey(20)

        if save and epoch % 10000 == 0:
            agent.save_model(directory, round_num, epoch)

        if agent.rewards == sum(np.array(auto_params[0])[:, 0] > 0):
            # FLAG THIS ONLY WORKS WITH ALL POSITIVE, 1 VALUED OBJECTS
            break
        # input()
    update_matrix = MC_matrix.copy()

    for index in range(epochs):
        # decaying value of rewards
        run_sum = 0
        for sum_range in range(index, epochs):
            run_sum += GAMMA**(sum_range - index) * MC_matrix[sum_range, 0]
        update_matrix[index, 0] = run_sum

    agent.mc_update(update_matrix, master_partials)


def run_trial(name, agent, rounds, epochs,
              given_params, save=True, animation=True):
    save_path = f"/Models/{name}/"
    if save:
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # p.dump(given_params, open(f"/Models/{name}/env.txt", "wb"))
        np.savetxt(f"/Models/{name}/env.txt", np.asarray(given_params[0]))
        with open(f"/Models/{name}/env_penalty.txt", "w") as f:
            f.write(f"Collision penalty is: {given_params[1]}")
        f.close()

    for round_num in range(rounds):
        world = Gridworld(WORLD_SIZE, ACTION_INFO, given_params)
        agent.set_world(world)
        world.place_agent(0, 0)
        run_epoch(
            agent,
            world,
            round_num,
            epochs,
            f"/Models/{name}",
            save=save,
            animate=animation)
        if save:
            agent.end_round(name, round_num)
        # if round_num / rounds > 0.25:
        #     agent.EPSILON = agent.EPSILON / 4


def test_agent(directory):
    agent = linear.Linear(
        0.1, 0.9, ACTION_INFO, WORLD_SIZE,
        weight_scheme="ZERO", learning_rate=0.01
    )
    agent.load_model(directory)
    return agent


def create_rand_env(world_dims, bombs):
    (x_size, y_size) = world_dims
    obj_list = np.zeros((bombs + 1, 3))
    out = []

    # sample coords so as not to overlap points
    coords = np.random.choice(
        x_size * y_size - 1,
        size=bombs + 1,
        replace=False)
    coords = coords + 1

    # set the cherry
    obj_list[0][0] = 1
    obj_list[0][1] = int(coords[0] % x_size)
    obj_list[0][2] = int(coords[0] // x_size)

    # set the bombs
    for bomb_obj in range(1, bombs + 1):
        obj_list[bomb_obj][0] = -1
        obj_list[bomb_obj][1] = int(coords[bomb_obj] % x_size)
        obj_list[bomb_obj][2] = int(coords[bomb_obj] // x_size)

    # rewrite in correct format
    obj_list = obj_list.astype(np.float).tolist()
    obj_list = [[element[0], True, element[1], element[2]]
                for element in obj_list]

    # and output
    out.append(obj_list)
    out.append(0)
    return out


if __name__ == "__main__":
    # agent = linear.Linear(
    #     0, 0.9, ACTION_INFO, WORLD_SIZE,
    #     weight_scheme="IDEAL", learning_rate=0.05
    # )
    # agent = linear.Linear(
    #     0.1, 0.9, ACTION_INFO, WORLD_SIZE,
    #     weight_scheme="RAND", learning_rate=0.05
    # )
    # agent = test_agent("/Models/tmp9/r50/modele99.txt")
    # world.load_world("/Models/tmp9/env.txt")
    # run_trial(RUN_NUM, agent, 100, 30, create_rand_env((WORLD_SIZE), 3), save=True)
    # print(create_rand_env((5, 10), 10))

    for trial in range(20):
        agent = linear.Linear(
            0.1, 0.9, ACTION_INFO, WORLD_SIZE,
            weight_scheme="RAND", learning_rate=0.005
        )

        run_trial(
            f"{RUN_NUM}_env{trial}",
            agent,
            100,
            30,
            create_rand_env((WORLD_SIZE), 1),
            save=True,
            animation=False)
