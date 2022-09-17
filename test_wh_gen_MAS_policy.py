import warehouse 
from warehouse.envs.warehouse import Warehouse
import gym
import ce
import random
import numpy as np
import random
from enum import Enum
import itertools

#
# Params
#
NUM_TASKS = 2
NUM_AGENTS = 2


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
init_agent_positions = [(0, 0), (9, 9)]
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
env.warehouse_api.set_random_task_rack(NUM_TASKS)
env.warehouse_api.set_random_task_feeds(NUM_TASKS)

print("random task racks: ", env.warehouse_api.task_racks)

obs = env.reset()
print("Initial observations: ", obs)
print("Agent rack positions: ", env.agent_rack_positions)

# ------------------------------------------------------------------------------
# Executor: Define a new executor which will be used to continually run agents
# ------------------------------------------------------------------------------

executor = ce.Executor(NUM_AGENTS)

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
for k in range(NUM_TASKS):
    mission.add_task(dfa)
# specify the storage outputs for the executor to access data accoss the MOTAP interface

# ------------------------------------------------------------------------------
# SCPM: Construct the SCPM structure which is a set of instructions on how to order the
# product MDPs
# ------------------------------------------------------------------------------

# Solve the product Model
scpm = ce.SCPM(mission, NUM_AGENTS, list(range(6)))
w = [0] * NUM_AGENTS + [1./ NUM_TASKS] * NUM_TASKS
eps = 0.00001
target = [-75, -56, 0.8, 0.8]
#
#
#
tnew = ce.scheduler_synth(scpm, env.warehouse_api, w, target, eps, executor)


## ------------------------------------------------------------------------------
## Execution: Construct a DFA transition function and build the Mission from this
## ------------------------------------------------------------------------------
#
agent_task_status = [AgentWorkingStatus.NOT_WORKING] * NUM_AGENTS
while True:
    actions = [6] * NUM_AGENTS
    for agent in range(NUM_AGENTS):
        if agent_task_status[agent] == AgentWorkingStatus.NOT_WORKING:
            task = executor.get_next_task(agent)
            env.agent_performing_task[agent] = task
            agent_task_status[agent] = AgentWorkingStatus.WORKING
        else:
            # Check if the agent's task has been completed
            pass
        # With the current task check what the dfa state is
        if env.agent_performing_task[agent] is not None:
            q = executor.dfa_current_state(env.agent_performing_task[agent])
            env.warehouse_api.set_task_(env.agent_performing_task[agent])
            #print("lookup state: ", env.get_state(agent), "task", agent_performings_task[agent], "q", q)
            actions[agent] = executor.get_action(agent, env.agent_performing_task[agent], env.states[agent], q)

            # step the agent forward one timestep
    obs, rewards, dones, info = env.step(actions)
    print("words", info)

    # Step the DFA forward
    for agent in range(NUM_AGENTS):
        current_task = env.agent_performing_task[agent]
        if current_task is not None:
            q = executor.dfa_current_state(current_task)
            executor.dfa_next_state(current_task, q, info[agent]["word"])
            qprime = executor.dfa_current_state(current_task)
            #print(f"Agent {0} state: {(obs[0]['a'], obs[0]['c'], env.agent_rack_positions[0])}"
            #      f"-> , action: {actions[agent]}, word: {info[agent]['word']}, Q': {qprime}")