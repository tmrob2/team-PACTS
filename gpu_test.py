import ce


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

scpm = ce.GPUSCPM(mission, NUM_AGENTS, list(range(2)))
w = [0., 1.]
eps = 0.00001

nz, nr, nc, i, p, x, csri, pi, rewards_vec, rewards_rows, rewards_cols = \
    ce.test_gpu_matrix(scpm, msg_env, w, eps)
NUM_MDPS = NUM_AGENTS * NUM_TASKS
block = nc * NUM_MDPS
total_length = block * 2

import scipy
import numpy as np

print(f"matrix size: {nr} x {nc}")
coo = scipy.sparse.coo_array((x, (i, p)), shape=(nr, nc))
csr = scipy.sparse.csr_array((x, p, csri), shape=(nr, nc))

# lets prototype the algorithm in scipy
csrtest = coo.tocsr().toarray()
csr = csr.toarray()
print("Conversion from COO to CSR succeeded?: ",(csrtest == csr).all())

# pick an initial policy
print("pi\n",list(map(lambda x: x[0] + nc * x[1], enumerate(pi))))
gather_idx = ce.map_policy_to_gather(pi, nc, total_length, block)
print("gather idx\n", gather_idx)
print("Testing GPU Dense gather")
output = [0] * 10
ce.test_thrust_gather_ffi(output)
rewards_ = np.array(rewards_vec).reshape(rewards_rows, rewards_cols)
print(rewards_)
nobjs = NUM_AGENTS + NUM_TASKS
gather_idx_rewards = ce.map_policy_to_gather(
    pi, 2, total_length, block
)
print("rewards gather idx:\n", gather_idx_rewards)




