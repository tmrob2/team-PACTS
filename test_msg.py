import ce


NUM_AGENTS = 100
NUM_TASKS = 100

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

dfa = message_sending_task(1)
#dfa.print_transitions()
mission.add_task(dfa)

scpm = ce.SCPM(mission, 1, list(range(2)))
w = [0., 1.]
eps = 0.00001

pi = ce.test_prod(scpm, msg_env, w, eps)

mission = ce.Mission()

for t in range(NUM_TASKS):
    dfa = message_sending_task(t + 1)
    mission.add_task(dfa)

scpm = ce.SCPM(mission, NUM_AGENTS, list(range(2)))

w = [0] * NUM_AGENTS + [1. / NUM_TASKS] * NUM_TASKS
eps = 0.0001
target = [-500.] * NUM_TASKS + [0.2] * NUM_TASKS

tnew = ce.msg_experiment(scpm, msg_env, w, target, eps)
