import ce
import random

NUM_AGENTS = 2
NUM_TASKS = 4


seed = 1234
# for this experiment we just want some identical copies of the agents
# define agent transitions

def transition_map(agent, seed, mu, std):
    # generate the transition with some random probability to simulate 
    # different agents
    random.seed(seed)
    p1 = random.normalvariate(mu, std)
    p2 = random.uniform(mu, std)
    agent.add_transition(0, 0, [(0, p1, ""), (1, 1-p1, "init")])
    agent.add_transition(1, 0, [(2, 1., "ready")])
    agent.add_transition(2, 0, [(3, 1-p2, "send"), (4, p2, "exit")])
    agent.add_transition(2, 1, [(4, 1.0, "exit")])
    agent.add_transition(3, 0, [(2, 1.0, "ready")])
    agent.add_transition(4, 0, [(0, 1.0, "")])
    return agent

def rewards_mdp(agent, r):
    # define rewards
    agent.add_reward(0, 0, -r)
    agent.add_reward(1, 0, -r)
    agent.add_reward(2, 0, -r)
    agent.add_reward(2, 1, -r)
    agent.add_reward(3, 0, -r)
    agent.add_reward(4, 0, -r)
    return agent

# Specify the agents
team = ce.Team()
for i in range(0, NUM_AGENTS):
    agent = ce.Agent(0, list(range(5)), [0, 1])
    if i == 1:
        agent = transition_map(agent, seed, 0.007, 0.001)
    else:
        agent = transition_map(agent, seed, 0.01, 0.001)
    agent = rewards_mdp(agent, 1)
    team.add_agent(agent.clone())

def construct_message_sending_task(r):
    task = ce.DFA(list(range(0, 6 + r)), 0, [2 + r + 1], [2 + r + 3], [2 + r + 2])
    for w in ["", "send", "ready", "exit"]:
        task.add_transition(0, w, 1)
    task.add_transition(0, "init", 2)
    task.add_transition(1, "init", 2)
    for w in ["", "send", "ready", "exit"]:
        task.add_transition(1, w, 1)
    for repeat in range(0, r + 1):
        task.add_transition(2 + repeat, "", 2 + repeat)
        task.add_transition(2 + repeat, "init", 2 + repeat)
        task.add_transition(2 + repeat, "ready", 2 + repeat)
        task.add_transition(2 + repeat, "send", 2 + repeat + 1)
        task.add_transition(2 + repeat, "exit", 2 + r + 3)
    for w in ["", "send", "ready", "exit", "init"]:
        task.add_transition(2 + r + 1, w, 2 + r + 2)
        task.add_transition(2 + r + 2, w, 2 + r + 2)
        task.add_transition(2 + r + 3, w, 2 + r + 3)
    return task

mission = ce.Mission()
words = ["", "send", "ready", "exit", "init"]
for k in range(0, NUM_TASKS):
    dfa = construct_message_sending_task(k)
    #print(f"DFA: {k} \n")
    #dfa.print_transitions(words)
    mission.add_task(dfa)

if __name__ == "__main__":
    #print(f"Rust calc sum: (5, 20) = {ce.sum_as_string(5, 20)}")
    #initial_state = (0, 0)
    # test creating the product mdp
    #print("Testing task 0")
    #product_mdp = ce.build_model(initial_state, agent, mission.get_task(3), 0, 3)
    #product_mdp.print_transitions()
    #product_mdp.print_rewards()
    
    scpm = ce.SCPM(team, mission)
    w = [0] * NUM_AGENTS + [1 / NUM_TASKS] * NUM_TASKS
    #ce.vi_test(product_mdp, w, NUM_AGENTS, NUM_TASKS)
    #w = [1 / (NUM_AGENTS + NUM_TASKS)] * ( NUM_AGENTS + NUM_TASKS )
    #w = [0, 0, 0.5, 0.5]
    #scpm.print_transitions()
    target = [-20., -20., 0.99, 0.97, 0.95, 0.95]
    ce.scheduler_synthesis(scpm, w, 0.0001, target)
    #ce.alloc_test(scpm, w, 0.0001)
