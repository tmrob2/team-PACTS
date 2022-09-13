from envs.warehouse import Warehouse
import warehouse
import gym
import ce
import random
import numpy as np
import random
import json
import logging
from enum import Enum
import itertools

# ------------------------------------------------------------------------------
# SETUP: Construct the structures for agent to recognise task progress
# ------------------------------------------------------------------------------

class TaskStatus: 
    INPROGESS = 0
    SUCCESS = 1
    FAIL = 2

class AgentWorkingStatus:
    NOT_WORKING = 0
    WORKING = 1

# Set the initial agent locations up front
# We can set the feed points up front as well because they are static
init_agent_positions = [(0, 0)]
size = 10
feedpoints = [(size - 1, size // 2)]
print("Feed points", feedpoints)

# ------------------------------------------------------------------------------
# Env Setup: Construct the warehouse model as an Open AI gym python environment
# ------------------------------------------------------------------------------
env: Warehouse = gym.make(
    "Warehouse-v0", 
    initial_agent_loc=init_agent_positions, 
    nagents=1,
    feedpoints=feedpoints,
    render_mode="human",
    size=size,
    seed=4321,
    disable_env_checker=True,
)
# We have to set the tasks racks and task feeds which depend on the number of tasks
env.warehouse_api.set_random_task_rack(1)
env.warehouse_api.set_random_task_feeds(1)

print("random task racks: ", env.warehouse_api.task_racks)

obs = env.reset()
print("Initial observations: ", obs)
print("Agent rack positions: ", env.agent_rack_positions)

# ------------------------------------------------------------------------------
# Tasks: Construct a DFA transition function and build the Mission from this
# ------------------------------------------------------------------------------

def warehouse_replenishment_task():
    task = ce.DFA(list(range(0, 8)), 0, [5], [6], [7])
    # attempt to goto the rack positon without carrying anything
    omega = set(env.warehouse_api.words)

    # The first transition determines if the label is at the rack
    task.add_transition(0, "R_NC", 1)
    excluded_words = ['_'.join(x) for x in list(itertools.product(["R", "NFR", "F"], ["P", "D", "C"]))]
    for w in excluded_words: 
        task.add_transition(0, f"{w}", 7)
    excluded_words.append("R_NC")
    for w in omega.difference(set(excluded_words)):
        task.add_transition(0, f"{w}", 0)
    # The second transition determines whether the agent picked up the rack at the 
    # required coord
    task.add_transition(1, "R_P", 2)
    excluded_words = ['_'.join(x) for x in list(itertools.product(["NFR"], ["P"]))]
    for w in excluded_words:
        task.add_transition(1, f"{w}", 7)
    excluded_words.append("R_P")
    for w in omega.difference(set(excluded_words)):
        task.add_transition(1, f"{w}", 1)
    # The third transition takes the agent to the feed position while carrying
    task.add_transition(2, "F_C", 3)
    excluded_words = ['_'.join(x) for x in list(itertools.product(["F", "R", "NFR"], ["NC", "P", "D"]))]
    for w in excluded_words:
        task.add_transition(2, f"{w}", 7)
    excluded_words.append("F_C")
    for w in omega.difference(set(excluded_words)):
        task.add_transition(2, f"{w}", 2)
    # The fourth transition takes the agent from the feed position while carrying 
    # back to the rack position
    task.add_transition(3, "R_C", 4)
    excluded_words = ['_'.join(x) for x in list(itertools.product(["F", "R", "NFR"], ["NC", "P", "D"]))]
    for w in excluded_words:
        task.add_transition(3, f"{w}", 7)
    excluded_words.append("R_C")
    for w in omega.difference(set(excluded_words)):
        task.add_transition(3, f"{w}", 3)
    # The fifth transition tells the agent to drop the rack at the required square
    task.add_transition(4, "R_D", 5)
    for w in omega.difference(set(["R_D"])):
        task.add_transition(4, f"{w}", 4)
    for w in omega:
        task.add_transition(5, f"{w}", 6)
    for w in omega:
        task.add_transition(6, f"{w}", 6)
    for w in omega:
        task.add_transition(7, f"{w}", 7)
    
    return task

# Initialise the mission
mission = ce.Mission()

# In this test there is only one task
dfa = warehouse_replenishment_task()
# Add the task to the mission
mission.add_task(dfa)
# specify the storage outputs for the executor to access data accoss the MOTAP interface
outputs = ce.MDPOutputs()

## ------------------------------------------------------------------------------
## SCPM: Construct the SCPM structure which is a set of instructions on how to order the
## product MDPs
## ------------------------------------------------------------------------------
#
## Solve the product Model
scpm = ce.SCPM(mission, 1, list(range(6)))
w = [0] * 1 + [1./ 1] * 1
eps = 0.00001
#
#
#
(pi, outputs) = ce.construct_prod_test(
    scpm, env.warehouse_api, w, eps, outputs
)
## ------------------------------------------------------------------------------
## Execution: Construct a DFA transition function and build the Mission from this
## ------------------------------------------------------------------------------
#
Q = [0] * 1
agent_task_status = AgentWorkingStatus.NOT_WORKING
agent_working_on_tasks = None 

# There is only once allocation, and it is (0, 0)

# Check from the mission whether it is complete or not
while not mission.check_mission_complete():
#for _ in range(100):
    # Specify that the agent is currently not working and that it may pick up a task
    actions = [-1]
    j = 0 # TODO how do we keep track of the task that the agent is working on
    print("lookup state: ", env.get_state(0))
    sidx = outputs.get_index(env.get_state(0), Q[0], 0, 0)
    #print(f"lookup state: s, {mdp_lookup_state}, q: {Q[0]}: sidx: {sidx}")
    actions[0] = int(pi[sidx])

    # Step the agent forward
    obs, rewards, dones, info = env.step(actions)

    # Now we need to get the next DFA step for both of the tasks
    # First get the word for the new observations 
    #print(f"New word for agent: {0} q, s': {obs[0]} is {word}")
    # step the task DFA forward
    qprime = mission.step(j, Q[j], info["word"])
    Q[j] = qprime#
    print(f"Agent {0} state: {(obs[0]['a'], obs[0]['c'], env.agent_rack_positions[0])} -> idx: {sidx}, action: {int(pi[sidx])}, word: {info['word']}, Q: {Q}")