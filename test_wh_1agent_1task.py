import warehouse 
from warehouse.envs.warehouse import Warehouse
import gym
import ce
import random
import numpy as np
import random
import json
import logging
from enum import Enum
import itertools

#
# Params
#
NUM_TASKS = 1
NUM_AGENTS = 1

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
    nagents=NUM_AGENTS,
    feedpoints=feedpoints,
    render_mode="human",
    size=size,
    seed=4321,
    disable_env_checker=True,
)
# We have to set the tasks racks and task feeds which depend on the number of tasks
rack_samples = random.sample([*env.warehouse_api.racks], k=2)
print("samples", rack_samples)
env.warehouse_api.add_task_rack_start(0, rack_samples[0])
env.warehouse_api.add_task_rack_end(0, rack_samples[1])
env.warehouse_api.add_task_feed(0, feedpoints[0])

print("random task racks: ", env.warehouse_api.task_racks_start)

obs = env.reset()
print("Initial observations: ", obs)
print("Agent rack positions: ", env.agent_rack_positions)

executor = ce.SerialisableExecutor(NUM_AGENTS)

# ------------------------------------------------------------------------------
# Tasks: Construct a DFA transition function and build the Mission from this
# ------------------------------------------------------------------------------

def warehouse_replenishment_task():
    task = ce.DFA(list(range(0, 8)), 0, [5], [7], [6])
    # attempt to goto the rack positon without carrying anything
    omega = set(env.warehouse_api.words)

    # The first transition determines if the label is at the rack
    task.add_transition(0, "RS_NC", 1)
    excluded_words = ['_'.join(x) for x in list(itertools.product(["RS", "RE", "NFR", "F"], ["P", "D", "CR", "CNR"]))]
    excluded_words.append("RE_NC")
    for w in excluded_words: 
        task.add_transition(0, f"{w}", 7)
    excluded_words.append("RS_NC")
    for w in omega.difference(set(excluded_words)):
        task.add_transition(0, f"{w}", 0)
    # The second transition determines whether the agent picked up the rack at the 
    # required coord
    task.add_transition(1, "RS_P", 2)
    excluded_words = ['_'.join(x) for x in list(itertools.product(["NFR"], ["P"]))]
    for w in excluded_words:
        task.add_transition(1, f"{w}", 7)
    excluded_words.append("RS_P")
    for w in omega.difference(set(excluded_words)):
        task.add_transition(1, f"{w}", 1)
    # The third transition takes the agent to the feed position while carrying
    task.add_transition(2, "F_CNR", 3)
    excluded_words = ['_'.join(x) for x in list(itertools.product(["F", "RS", "RE", "NFR"], ["NC", "P", "D", "CR"]))]
    for w in excluded_words:
        task.add_transition(2, f"{w}", 7)
    excluded_words.append("F_CNR")
    for w in omega.difference(set(excluded_words)):
        task.add_transition(2, f"{w}", 2)
    # The fourth transition takes the agent from the feed position while carrying 
    # back to the rack position
    task.add_transition(3, "RS_CNR", 4)
    excluded_words = ['_'.join(x) for x in list(itertools.product(["F", "RS", "RE", "NFR"], ["NC", "P", "D", "CR"]))]
    #excluded_words.append("RS_CNR")
    for w in excluded_words:
        task.add_transition(3, f"{w}", 7)
    excluded_words.append("RS_CNR")
    for w in omega.difference(set(excluded_words)):
        task.add_transition(3, f"{w}", 3)
    # The fifth transition tells the agent to drop the rack at the required square
    task.add_transition(4, "RS_D", 5)
    for w in omega.difference(set(["RS_D"])):
        task.add_transition(4, f"{w}", 4)
    for w in omega:
        task.add_transition(5, f"{w}", 6)
    for w in omega:
        task.add_transition(6, f"{w}", 6)
    for w in omega:
        task.add_transition(7, f"{w}", 7)
    
    return task

def warehouse_retry_task():
    task = ce.DFA(list(range(0, 8)), 0, [5], [7], [6])
    # attempt to goto the rack positon without carrying anything
    omega = set(env.warehouse_api.words)

    # The first transition determines if the label is at the rack
    task.add_transition(0, "RS_NC", 1)
    excluded_words = ['_'.join(x) for x in list(itertools.product(["RS", "RE", "NFR", "F"], ["P", "D", "CR", "CNR"]))]
    excluded_words.append("RE_NC")
    for w in excluded_words: 
        task.add_transition(0, f"{w}", 7)
    excluded_words.append("RS_NC")
    for w in omega.difference(set(excluded_words)):
        task.add_transition(0, f"{w}", 0)
    # The second transition determines whether the agent picked up the rack at the 
    # required coord
    task.add_transition(1, "RS_P", 2)
    excluded_words = ['_'.join(x) for x in list(itertools.product(["NFR"], ["P"]))]
    for w in excluded_words:
        task.add_transition(1, f"{w}", 7)
    excluded_words.append("RS_P")
    for w in omega.difference(set(excluded_words)):
        task.add_transition(1, f"{w}", 1)
    # The third transition takes the agent to the feed position while carrying
    task.add_transition(2, "F_CNR", 3)
    excluded_words = ['_'.join(x) for x in list(itertools.product(["F", "RS", "RE", "NFR"], ["NC", "P", "D", "CR"]))]
    for w in excluded_words:
        task.add_transition(2, f"{w}", 7)
    excluded_words.append("F_CNR")
    for w in omega.difference(set(excluded_words)):
        task.add_transition(2, f"{w}", 2)
    # The fourth transition takes the agent from the feed position while carrying 
    # back to the rack position
    task.add_transition(3, "RE_CNR", 4)
    excluded_words = ['_'.join(x) for x in list(itertools.product(["F", "RS", "RE", "NFR"], ["NC", "P", "D", "CR"]))]
    #excluded_words.append("RS_CNR")
    for w in excluded_words:
        task.add_transition(3, f"{w}", 7)
    excluded_words.append("RE_CNR")
    for w in omega.difference(set(excluded_words)):
        task.add_transition(3, f"{w}", 3)
    # The fifth transition tells the agent to drop the rack at the required square
    task.add_transition(4, "RE_D", 5)
    for w in omega.difference(set(["RE_D"])):
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
dfa_replen = warehouse_replenishment_task()
dfa_retry = warehouse_retry_task()
# Add the task to the mission
mission.add_task(dfa_retry)

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
pi = ce.construct_prod_test(scpm, env.warehouse_api, w, eps)
## ------------------------------------------------------------------------------
## Execution: Construct a DFA transition function and build the Mission from this
## ------------------------------------------------------------------------------