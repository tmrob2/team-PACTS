from email.mime import base
import redis
import warehouse 
from warehouse.envs.warehouse import Warehouse
import ce
import argparse
import gym
import itertools
import json

# TODO set these with argparse
batch_size = 5
num_agents = 3

# create a subscriber to listen to messages
r = redis.Redis(host='localhost', port=6379, db=0)

# ------------------------------------------------------------------------------
# Env Setup: Construct the warehouse model as an Open AI gym python environment
# ------------------------------------------------------------------------------
init_agent_positions = [(0, 0), (4, 0), (0, 4)]
size = 10

feedpoints = [(size - 1, size // 2)]
env: Warehouse = gym.make(
    "Warehouse-v0", 
    initial_agent_loc=init_agent_positions, 
    nagents=num_agents,
    feedpoints=feedpoints,
    render_mode="human",
    size=size,
    seed=4321,
    disable_env_checker=True,
)

p = r.pubsub()
p.subscribe('executor-channel')

task_progress = {0: "initial", 1: "in_progress", 2: "success", 3: "fail"}

# construct a new executor
base_exec = ce.Executor(num_agents)
env.reset()

while True:
    message = p.get_message() # this is a redis framework method
    # we need to collect a batch either by time interval or fixed count of 
    # messages
    if message:
        print(message.keys())
        # message is a string and we want to try and convert it to a task
        try:
            # TODO we need a mapping between the batch count and the actual task index
            data_ = message["data"].decode('utf-8')
            #print("data_", data_)
            data = json.loads(data_)
            print(data.keys())
            if data['event_type'] == "execute":
                print("received execute event")
                # A serialised executor comes through the channel

                # Convert the serialised executor to an Executor

                # Merge the Executor with this main thread running executor
                # -> this is because the pygame/env will consume the main executor
                #    so we need to add to this only
                serialised_executor = ce.SerialisableExecutor(None, data['executor'])
                # The new task feeds and task racks should be inserted into the warehouse
                # environment according to the task mapping
                print(data['task-feed'])
                print(data['task-rack'])
                raw_task_feed = data['task-feed']
                raw_task_rack = data['task-rack']
                    
                executor = serialised_executor.convert_to_executor(num_agents, batch_size)
                for k in range(batch_size):
                    task_idx = executor.get_task_mapping(k)
                    env.warehouse_api.add_task_rack_start(task_idx, tuple(raw_task_rack[str(k)]))
                    env.warehouse_api.add_task_feed(task_idx, tuple(raw_task_feed[str(k)]))
                print(executor.agent_task_allocations)
                base_exec.merge_exec(executor, num_agents, batch_size)
                print("alloc after merge\n", base_exec.agent_task_allocations)
            elif data['event_type'] == 'end_stream':
                # send a poison pill to the thread
                print("received tear down event, ending listener")
                break
            # construct and SCPM to process the tasks
        except:
            print(f"Unable to decode msg")

    actions = [6] * num_agents
    for agent in range(num_agents):
        if env.agent_task_status[agent] == env.AgentWorkingStatus.NOT_WORKING:
            task = base_exec.get_next_task(agent)
            env.agent_performing_task[agent] = task
            env.states[agent] = (env.states[agent][0], 0, None)
            if task is not None:
                env.agent_task_status[agent] = env.AgentWorkingStatus.WORKING
            if task is not None:
                print("rack: ", env.warehouse_api.task_racks_start[task])
        else:
            # Check if the agent's task has been completed
            if env.agent_performing_task[agent] is not None:
                status = base_exec.check_done(env.agent_performing_task[agent])
                #print("task: ", env.agent_performing_task[agent], "status: (", status, task_progress[status], ")")
                if task_progress[status] in ["success", "fail"]:
                    print(f"Task {env.agent_performing_task[agent]} -> {task_progress[status]}")
                    # goto the the next task
                    if task_progress[status] == "fail":
                        print("orig rack position", env.orig_rack_positions[env.agent_performing_task[agent]])
                        print("current rack position", env.agent_rack_positions[agent])
                        r.publish('executor-channel', json.dumps({
                            'event_type': "task_failure",
                            'rack_start': env.orig_rack_positions[env.agent_performing_task[agent]],
                            'current_rack_position': env.agent_rack_positions[agent]
                        }))
                    env.agent_task_status[agent] = env.AgentWorkingStatus.NOT_WORKING
                    env.agent_rack_positions[agent] = None
                    # todo add cleanup method here to remove the completed tasks from memory

        # With the current task check what the dfa state is
        if env.agent_performing_task[agent] is not None:
            q = base_exec.dfa_current_state(env.agent_performing_task[agent])
            env.warehouse_api.set_task_(env.agent_performing_task[agent])
            #print("lookup state: ", env.get_state(agent), "task", agent_performings_task[agent], "q", q)
            actions[agent] = base_exec.get_action(agent, env.agent_performing_task[agent], env.states[agent], q)

            # step the agent forward one timestep
    #print(env.agent_performing_task)
    obs, rewards, dones, info = env.step(actions)
    #print("words", info)

    # Step the DFA forward
    for agent in range(num_agents):
        current_task = env.agent_performing_task[agent]
        if current_task is not None:
            q = base_exec.dfa_current_state(current_task)
            base_exec.dfa_next_state(current_task, q, info[agent]["word"])
            qprime = base_exec.dfa_current_state(current_task)
            #print(f"Agent {0} state: {(obs[0]['a'], obs[0]['c'], env.agent_rack_positions[0])}"
            #      f"-> , action: {actions[agent]}, word: {info[agent]['word']}, Q': {qprime}")

