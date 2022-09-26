import ce
import random
import numpy as np
import random

NUM_AGENTS = 2
NUM_TASKS = 2

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
    target = [-50, -50] + [0.8] * NUM_TASKS
    eps = 0.00001
    init_agent_positions = [(0, 0), (9, 9)]
    ce.construct_prod_test(scpm, 10, 10, w, eps)
    ce.test_scpm(scpm, w, eps, 10, 10, init_agent_positions)
    
