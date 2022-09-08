import warehouse
import gym
import ce
import random
import numpy as np
import random
import json
import logging
from enum import Enum

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
xsize, ysize = 10, 10
feedpoints = [(xsize - 1, ysize // 2)]

# Place the racks in the warehouse, we do this with a rust interface because
# the warehouse can be quite large and it is O(n^2)
racks = ce.place_racks(xsize, ysize)

# ------------------------------------------------------------------------------
# Env Setup: Construct the warehouse model as an Open AI gym python environment
# ------------------------------------------------------------------------------
env = gym.make(
    "Warehouse-v0", 
    initial_agent_loc=init_agent_positions, 
    nagents=1,
    feedpoints=feedpoints,
    render_mode="human",
    xsize=xsize,
    ysize=ysize,
    disable_env_checker=True
)

obs = env.reset()
print("Initial observations: ", obs)
print("Agent rack positions: ", env.agent_rack_position)

# ------------------------------------------------------------------------------
# Tasks: Construct a DFA transition function and build the Mission from this
# ------------------------------------------------------------------------------
words = set(["a", "fail", "na", "r", "nr", "cf", "ncf", "ca", "nca", "er", "true"])

def warehouse_replenishment_task():
    task = ce.DFA(list(range(0, 8)), 0, [5], [6], [7])
    # attempt to goto the feed positon without carrying anything
    task.add_transition(0, "a", 1)
    task.add_transition(0, "fail", 7)
    for w in list(words.difference(set(["a", "fail"]))):
        task.add_transition(0, w, 0)
    # pick up the rack at pos a
    task.add_transition(1, "r", 2)
    for w in list(words.difference(set("r"))):
        task.add_transition(1, w, 1)
    # Once the rack has been picked up carry it over to its feed location
    task.add_transition(2, "cf", 3)
    task.add_transition(2, "fail", 7)
    for w in list(words.difference(set(["cf", "fail"]))):
        task.add_transition(2, w, 2)
    # carry the rack back to the rack position
    task.add_transition(3, "ca", 4)
    task.add_transition(3, "fail", 7)
    for w in list(words.difference(set(["ca", "fail"]))):
        task.add_transition(3, w, 3)
    task.add_transition(4, "er", 5)
    task.add_transition(4, "ca", 4)
    task.add_transition(5, "true", 6)
    task.add_transition(6, "true", 6)
    task.add_transition(7, "true", 7)
    return task

# Initialise the mission
mission = ce.Mission()

# In this test there is only one task
dfa = warehouse_replenishment_task()
# Add the task to the mission
mission.add_task(dfa)
# specify the storage outputs for the executor to access data accoss the MOTAP interface
outputs = ce.MDPOutputs()

# ------------------------------------------------------------------------------
# SCPM: Construct the SCPM structure which is a set of instructions on how to order the
# product MDPs
# ------------------------------------------------------------------------------

# Solve the product Model
scpm = ce.SCPM(mission, 1, list(range(6)))
w = [0] * 1 + [1./ 1] * 1
eps = 0.00001

rack_tasks = random.choices(
    list(racks), weights=[1/len(racks)] * len(racks), k=1
)
print("Rack tasks: ", rack_tasks)
task_feeds = random.choices(
    list(feedpoints), weights=[1/len(feedpoints)] * len(feedpoints), k=1
)

(pi, outputs) = ce.construct_prod_test(
    scpm, xsize, ysize, w, eps, init_agent_positions[0], rack_tasks, task_feeds, feedpoints, outputs
)
# ------------------------------------------------------------------------------
# Execution: Construct a DFA transition function and build the Mission from this
# ------------------------------------------------------------------------------

Q = [0] * 1
agent_task_status = AgentWorkingStatus.NOT_WORKING
agent_working_on_tasks = None 
# There is only once allocation, and it is (0, 0)

# Check from the mission whether it is complete or not
while not mission.check_mission_complete():
    # Specify that the agent is currently not working and that it may pick up a task
    actions = [-1]

    # We need something like current tasks to keep track of the tasks being executed

    # There is a problem here because I don't think that outputs can be serialized
    # but the executor needs to run on another thread
    mdp_lookup_state = (tuple(obs[0]["a"]), obs[0]["c"], env.agent_rack_position[0])
    #print("lookup state", mdp_lookup_state)
    sidx = outputs.get_index(mdp_lookup_state, Q[0], 0, 0)
    #print(f"Agent {0} state: {(obs[0]['a'], obs[0]['c'], env.agent_rack_position[0])} -> idx: {sidx}, action: {int(pi[sidx])}")
    actions[0] = int(pi[sidx])

    # Step the agent forward
    obs, rewards, dones, info = env.step(actions)
    #print("New Obs: ", obs)

    # Now we need to get the next DFA step for both of the tasks
    # First get the word for the new observations 
    j = 0
    word = env.label(Q[j], 0, rack_tasks, j, task_feeds[j])
    #print(f"New word for agent: {0} q, s': {obs[0]} is {word}")
    # step the task DFA forward
    qprime = mission.step(j, Q[j], word)
    Q[j] = qprime