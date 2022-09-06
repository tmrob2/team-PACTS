import warehouse
import gym
import ce
import random
import numpy as np
import random
import json

NUM_AGENTS = 2
NUM_TASKS = 2

# Set the initial agent locations up front
# We can set the feed points up front as well because they are static
init_agent_positions = [(0, 0), (9, 9)]
xsize, ysize = 10, 10
feedpoints = [(xsize - 1, ysize // 2)]
racks = ce.place_racks(xsize, ysize)
env = gym.make(
    "Warehouse-v0", 
    initial_agent_loc=init_agent_positions, 
    nagents=2,
    feedpoints=feedpoints,
    render_mode="human",
    xsize=xsize,
    ysize=ysize
)

obs = env.reset()
print("Initial observations: ", obs)
print("Agent rack positions: ", env.agent_rack_position)

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

# todo: in the future the warehouse data should probably be set in the 
# python script
mission = ce.Mission()

for k in range(0, NUM_TASKS):
    dfa = warehouse_replenishment_task()
    #print(f"DFA: {k} \n")
    #dfa.print_transitions(words)
    mission.add_task(dfa)

if __name__ == "__main__":
    scpm = ce.SCPM(mission, NUM_AGENTS, list(range(6)))
    w = [0] * NUM_AGENTS + [1./ NUM_TASKS] * NUM_TASKS
    target = [-130, -50] + [0.8] * NUM_TASKS
    eps = 0.00001
    init_agent_positions = [(0, 0), (9, 9)]
    #1. - ce.construct_prod_test(scpm, 13, 13, w, eps)
    #ce.test_scpm(scpm, w, eps, 10, 10, )
    # generate the rack positions using the MOTAP interface
    # generate a set of rack tasks equal to the size of the num of tasks
    rack_tasks = random.choices(
        list(racks), weights=[1/len(racks)] * len(racks), k=NUM_TASKS
    )
    task_feeds = random.choices(
        list(feedpoints), weights=[1/len(feedpoints)] * len(feedpoints), k=NUM_TASKS
    )

    outputs = ce.MDPOutputs()
    complete = False
    while not complete:
        try:
            weights, l, pis, outputs = ce.scheduler_synthesis(
                scpm, w, eps, target, 10, 10, init_agent_positions, task_feeds, racks, rack_tasks, outputs
            )
            complete = True

            # for each of the agents in the team we require an allocation

            # so starting at the initial state
            # render the solution while keeping track of the DFA
        except Exception as e:
            print("weights not returned: \n", e)

    weights_ = np.reshape(weights, (NUM_TASKS, l))
    print(list(weights_))
    # Now we need to select scehdulers for tasks according to the marginals
    # then put this into the environment and do a rendering
    # after this we can make the process continuous
    task_alloc = {i: [] for i in range(NUM_AGENTS)}
    for task in range(NUM_TASKS):
        print("task 0 weights", weights_[task])
        ind0 = random.choices(list(range(l)), weights_[task])
        print(f"choose scheduler {ind0} for task {task}")
        d_ = pis[ind0[0]]
        agent_task = list(filter(lambda x: x[1] == task, d_.keys()))
        print(f"j{agent_task[0][1]} -> i{agent_task[0][0]}")
        task_alloc[agent_task[0][0]].append((agent_task[0][1], ind0[0]))
    print("task allocation\n", task_alloc)
    #actions = [env.action_space.sample() for k in range(NUM_AGENTS)]
    #print("action: ", actions)
    #obs, reward, dones, info = env.step(actions)
    # given some word from the observation
    #env.render()
    # starting at the initial state, which is already built into the agent
    # get the first action from the policy
    #for agent in range(NUM_AGENTS):
    # pop the first task from the allocation
    tasks = {i: None for i in range(NUM_AGENTS)}
    for i in range(NUM_AGENTS):
        if task_alloc[i]:
            tasks[i] = task_alloc[i].pop()

    print("tasks: ", tasks)
    for idx in range(NUM_AGENTS):
        sidx = outputs.get_index((obs[idx]["a"], obs[idx]["c"], env.agent_rack_position[idx]))
        print("sidx", sidx)
        # get the index from the new state
        print(f"task {idx} -> {tasks[i]}")
        actions = [-1, -1]
        if tasks[idx]:
            k = tasks[idx][1]
            j = tasks[idx][0]
            pi = pis[k][(idx, j)]
            print(f"Agent {idx} state: {(obs[idx]['a'], obs[idx]['c'], env.agent_rack_position[idx])} -> idx: {sidx}, action: {int(pi[sidx])}")
            actions[idx] = int(pi[sidx])
        else:
            print(f"Agent {idx} state: {(obs[idx]['a'], obs[idx]['c'], env.agent_rack_position[idx])} -> idx: {sidx}, action: {6}")
            actions[idx] = 6
        obs, rewards, dones, info = env.step(actions)
        
        