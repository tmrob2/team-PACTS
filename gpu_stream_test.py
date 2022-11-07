import ce

ce.device_props()

# construct a model and an environment

NUM_AGENTS = 1
NUM_TASKS = 1

msg_env = ce.MessageSender()

words = ["", "i", "r", "e", "s"]

def message_sending_task(num_msgs):
    task = ce.DFA(
        list(range(1 + num_msgs + 3)), 
        0, 
        [num_msgs + 1], 
        [num_msgs + 3], 
        [num_msgs + 2]
    )
    task.add_transition(0, "i", 1)
    for w in set(words).difference(set(["i"])):
        task.add_transition(0, w, 0)
    for r in range(num_msgs):
        task.add_transition(1 + r, "s", 1 + r + 1)
        task.add_transition(1 + r, "e", num_msgs + 3)
        for w in set(words).difference(set(["s", "e"])):
            task.add_transition(1 + r, w, 1 + r)
    for w in words:
        task.add_transition(num_msgs + 1, w, num_msgs + 2)
        task.add_transition(num_msgs + 2, w, num_msgs + 2)
        task.add_transition(num_msgs + 3, w, num_msgs + 3)
    return task

mission = ce.Mission()

for repeat in range(NUM_TASKS):
    dfa = message_sending_task(repeat + 1)
    mission.add_task(dfa)

scpm = ce.GPUSCPMCompact(mission, NUM_AGENTS, list(range(2)))
w = [0., 1.]
eps = 0.00001

ce.test_construct_compact_gpu(scpm, msg_env)