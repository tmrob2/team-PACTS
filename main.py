import ce

NUM_AGENTS = 2
NUM_TASKS = 4

agent = ce.Agent(0, list(range(5)), [0, 1])
# for this experiment we just want some identical copies of the agents
# define agent transitions
agent.add_transition(0, 0, [(0, 0.01, ""), (1, 0.99, "init")])
agent.add_transition(1, 0, [(2, 1., "ready")])
agent.add_transition(2, 0, [(3, 0.99, "send"), (4, 0.01, "exit")])
agent.add_transition(2, 1, [(4, 1.0, "exit")])
agent.add_transition(3, 0, [(2, 1.0, "ready")])
agent.add_transition(4, 0, [(0, 1.0, "")])
# define rewards
agent.add_reward(0, 0, -1)
agent.add_reward(1, 0, -1)
agent.add_reward(2, 0, -1)
agent.add_reward(2, 1, -1)
agent.add_reward(3, 0, -1)
agent.add_reward(4, 0, -1)

# Specify the agents
team = ce.Team()
for i in range(0, NUM_AGENTS):
    team.add_agent(agent.clone())

def construct_message_sending_task(r):
    task = ce.DFA(list(range(0, 6)), 0, [2 + r + 1], [2 + r + 3], [2 + r + 2])
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
for k in range(0, NUM_TASKS):
    mission.add_task(construct_message_sending_task(k))

if __name__ == "__main__":
    #print(f"Rust calc sum: (5, 20) = {ce.sum_as_string(5, 20)}")
    initial_state = (0, 0)
    # test creating the product mdp
    product_mdp = ce.build_model(initial_state, agent, mission.get_task(0), 0, 0)
    product_mdp.print_transitions()
    product_mdp.print_rewards()
    ce.vi_test(product_mdp)
    #scpm = ce.SCPM(team, mission)
    #ce.value_iteration(scpm)
