from pyexpat import model
from re import M
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

P, pi, R, data = ce.test_gpu_matrix(scpm, msg_env)
NUM_MDPS = NUM_AGENTS * NUM_TASKS

import scipy
import numpy as np
np.set_printoptions(suppress=True)
coo_test = scipy.sparse.coo_array((P.x, (P.i, P.p)), shape=(P.m, P.n))
csr_test = coo_test.tocsr().toarray()

#Pcsr = ce.test_compress(P)
Pcsr = ce.compress(P)
Rcsr = ce.compress(R)
#print("i", list(enumerate(Pcsr.i)))
#print("j", Pcsr.p)
#print("x", Pcsr.x)
csr = scipy.sparse.csr_array((Pcsr.x, Pcsr.p, Pcsr.i), shape=(Pcsr.m, Pcsr.n)).toarray();

#print("\Transition matrix")
#for r in range(Pcsr.m):
#    print("|", end=" ")
#    for c in range(Pcsr.n):
#        if c < Pcsr.n - 1:
#            print(f"{csr[r, c]:.2f}", end=" ")
#        else:
#            print(f"{csr[r, c]:.2f} |")

print("CSR construction correct", (csr == csr_test).all())

print("Rewards Matrix:\n", scipy.sparse.csr_array((Rcsr.x, Rcsr.p, Rcsr.i), shape=(Rcsr.m, Rcsr.n)).toarray())

pitest = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
argmaxP = ce.test_argmax_csr(
    Pcsr, 
    pitest, 
    data.transition_prod_block_size, 
    data.transition_prod_block_size
)
argmaxP = scipy.sparse.csr_array(
    (argmaxP.x, argmaxP.p, argmaxP.i), 
    shape=(data.transition_prod_block_size, data.transition_prod_block_size)
).toarray()
print("\nArgmax Transition matrix @ test policy", pitest)
for r in range(data.transition_prod_block_size):
    print("|", end=" ")
    for c in range(data.transition_prod_block_size):
        if c < data.transition_prod_block_size - 1:
            print(f"{argmaxP[r, c]:.2f}", end=" ")
        else:
            print(f"{argmaxP[r, c]:.2f} |")

argmaxR = ce.test_argmax_csr(
    Rcsr, 
    pitest, 
    data.transition_prod_block_size, 
    data.reward_obj_prod_block_size
)

argmaxR = scipy.sparse.csr_array(
    (argmaxR.x, argmaxR.p, argmaxR.i),
    shape=(data.transition_prod_block_size, data.reward_obj_prod_block_size)
).toarray()

print("\nArgmax Rewards matrix @ test policy: ", np.array(pitest))
for r in range(data.transition_prod_block_size):
    print("|", end=" ")
    for c in range(data.reward_obj_prod_block_size):
        if c < data.reward_obj_prod_block_size - 1:
            print(f"{argmaxR[r, c]:.2f}", end=" ")
        else:
            print(f"{argmaxR[r, c]:.2f} |")

ce.gpu_test_csr_mv()

# compare the rewards matrix with scipy for csr compression

rmv = [0.] * data.transition_prod_block_size
# fix an initial policy and make sure that the GPU and CPU results agree

cpu_scpm = ce.SCPM(mission, 1, list(range(2)))
w_test = [1.0, 0.]
eps = 0.00001
cpu_test_val = ce.test_cpu_init_pi(cpu_scpm,msg_env, pitest)
print("TEST: CPU step 0 val for test policy\n", cpu_test_val)

print("Trans block size: ", data.transition_prod_block_size)
print("Reward block size: ", data.reward_obj_prod_block_size)

vmv = ce.test_initial_policy(
    pitest, 
    Pcsr, 
    Rcsr, 
    data.transition_prod_block_size,
    data.transition_prod_block_size,
    data.reward_obj_prod_block_size,
    w_test,
    rmv,
    eps
)

testval = ce.test_cpu_converged_init_pi(
    cpu_scpm,
    msg_env,
    pitest,
    eps,
    0, 
    0
)

print("TEST: VALUE GPU == CPU", 
    "SUCCEEDED" if (np.round(np.array(vmv), 2) == np.round(np.array(testval), 2)).all() \
        else "FAILED")

print("TEST: Output from CPU init value for test policy\n", testval)

(i, j, x, nz, nc, nr) = ce.test_output_trans_matrix(
    cpu_scpm,
    msg_env,
    pitest
)

test_csc = scipy.sparse.csc_matrix((x, i, j), shape=(nr, nc)).toarray()
#print(test_csc)


print("TEST: CSR==CSS =>", "SUCCEEDED" if (test_csc == np.round(argmaxP, 2)).all() else "FAILED")

cpu_rewards, rnc, rnr = ce.test_output_rewards_matrix(
    cpu_scpm,
    msg_env,
    pitest,
    2
)

test_rewards = np.array(cpu_rewards).reshape(rnr, rnc, order="F")
#print(test_rewards)


print("TEST: VALUE ITERATION")
ce.test_gpu_value_iteration(
    scpm,
    msg_env,
    [0.5, 0.5],
    0.0001
)
print("TEST: END\n\n")
mission = ce.Mission()
for repeat in range(2):
    dfa = message_sending_task(repeat + 1)
    mission.add_task(dfa)

scpm = ce.GPUSCPM(mission, 2, list(range(2)))
wgpu = [1.0, 1.0, 0.0, 0.0] * 2 * 2
eps = 0.00001

P, pi, R, data = ce.test_gpu_matrix(scpm, msg_env)


Pcsr = ce.compress(P)
Rcsr = ce.compress(R)
rmv = [0.] * data.transition_prod_block_size

print("pi", np.array(pi))
pitest2 = [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 
           0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 
           0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 
           0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 
           0, 1, 0, 0, 0, 0, 0, 0, 0, 1,
           0, 0, 0, 0, 1, 0, 0, 0, 0, 0]

#pitest2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
#           0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 
#           0, 0, 0, 1, 0, 0, 0, 0, 0, 0]

argmaxP = ce.test_argmax_csr(
    Pcsr, 
    pitest2, 
    data.transition_prod_block_size, 
    data.transition_prod_block_size
)

argmaxP = scipy.sparse.csr_array(
    (argmaxP.x, argmaxP.p, argmaxP.i),
    shape=(data.transition_prod_block_size, data.transition_prod_block_size)
).toarray()

#print("transition block size", data.transition_prod_block_size)
#print("reward block size", data.reward_obj_prod_block_size)
#
#print("\nArgmax Trans matrix @ test policy: \n",pitest2)
#for r in range(data.transition_prod_block_size):
#    print("|", end=" ")
#    for c in range(data.transition_prod_block_size):
#        if c < data.transition_prod_block_size - 1:
#            print(f"{argmaxP[r, c]:.2f}", end=" ")
#        else:
#            print(f"{argmaxP[r, c]:.2f} |")

a2t2gpuvalue = ce.test_initial_policy(
    pitest2, 
    Pcsr, 
    Rcsr, 
    data.transition_prod_block_size,
    data.transition_prod_block_size,
    data.reward_obj_prod_block_size,
    wgpu,
    rmv,
    eps
)

cpu_scpm = ce.SCPM(mission, 2, list(range(2)))
gpu_state_spaces = scpm.state_spaces;
print("GPU STATE SPACES", gpu_state_spaces)
total_val = []
cntr = 0
state_size = 0
for agent in range(2):
    for task in range(2):
        current_size = gpu_state_spaces[cntr]
        current_pi = pitest2[state_size:state_size + current_size]
        print("pi: ", current_pi)
        testval = ce.test_cpu_converged_init_pi(
            cpu_scpm,
            msg_env,
            current_pi,
            eps,
            agent,
            task
        )
        #print("test val new:", testval)
        total_val.extend(testval)
        state_size += current_size
        cntr += 1
#
print("TEST: A2T2 GPU == CPU", 
    "SUCCEEDED" if (np.round(np.array(a2t2gpuvalue)) \
        == np.round(np.array(total_val))).all() else "FAILED"
)

opt_pi = ce.test_policy_optimisation(
    a2t2gpuvalue,
    pitest2,
    Pcsr,
    Rcsr,
    [0.5, 0.5, 0.0, 0.0],
    0.0001,
    2,
    4,
    data.transition_prod_block_size
)

NobjSparse = ce.test_nobj_argmax_csr(
    Pcsr,
    opt_pi,
    data.transition_prod_block_size,
    data.transition_prod_block_size,
    1
)

print("m:", NobjSparse.m)
print("n", NobjSparse.n)
print("nz", NobjSparse.nz)

ce.test_gpu_value_iteration(
    scpm,
    msg_env,
    [0., 0., 0.5, 0.5],
    0.000001 
)

mission = ce.Mission()
for repeat in range(100):
    dfa = message_sending_task(repeat + 1)
    mission.add_task(dfa)

scpm = ce.GPUSCPM(mission, 2, list(range(2)))

#ce.test_gpu_value_iteration(
#    scpm,
#    msg_env,
#    [0., 0., 1/3, 1/3, 1/3],
#    0.000001 
#   )

#w = [0., 0., 1/3, 1/3, 1/3]
w = [0.] * 2 + [1./10.] * 100
t = [-50, -50] + [0.5] * 100
eps = 0.00001
ce.test_gpu_synth(
    scpm, 
    msg_env, 
    w,
    eps,
    t
)
