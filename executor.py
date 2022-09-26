from curses import raw
from email.mime import base
import redis
import warehouse 
from warehouse.envs.warehouse import Warehouse
import ce
import argparse
import gym
import itertools
import json
from utils.gym_to_gif import save_frames_as_gif

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
    render_mode="rgb_array",
    size=size,
    seed=4321,
    prob=0.99,
    disable_env_checker=True,
)

p = r.pubsub()
p.subscribe('executor-channel')

task_progress = {0: "initial", 1: "in_progress", 2: "success", 3: "fail"}

# construct a new executor
base_exec = ce.Executor(num_agents)
env.reset()

frames = []


while True:
    message = p.get_message()
    # we need to collect a batch either by time interval or fixed count of 
    # messages
    if message:
        #print(message.keys())
        #print(message)
        # message is a string and we want to try and convert it to a task
        if message['channel'] == b'executor-channel' \
            and not isinstance(message['data'], int):
            data_ = message["data"].decode('utf-8')
            #print("data_", data_)
            data = json.loads(data_)
            #print(data.keys())
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
                raw_task_feed = data['task-feed']
                raw_task_rack_start = data['task-rack-start']
                raw_task_rack_end = data["task-rack-end"]
                batch_size_ = data["batch-size"]
                
                    
                executor = serialised_executor.convert_to_executor(num_agents, batch_size_)
                for k in range(batch_size_):
                    task_idx, _ = executor.get_task_mapping(k)
                    env.warehouse_api.add_task_rack_start(task_idx, tuple(raw_task_rack_start[str(k)]))
                    env.warehouse_api.add_task_feed(task_idx, tuple(raw_task_feed[str(k)]))
                    if str(k) in raw_task_rack_end.keys():
                        env.warehouse_api.add_task_rack_end(task_idx, tuple(raw_task_rack_end[str(k)]))
                        # this is a failure task i.e. repeat
                        env.failure_task_progress[task_idx] = {
                            "original_position": tuple(raw_task_rack_end[str(k)]),
                            "start_position": tuple(raw_task_rack_start[str(k)])
                        }

                base_exec.merge_exec(executor, num_agents, batch_size_)
                print("alloc after merge\n", base_exec.agent_task_allocations)
            elif data['event_type'] == 'end_stream':
                # send a poison pill to the thread
                print("received tear down event, ending listener")
                break
            # construct and SCPM to process the tasks

    # Construct a randering of the solution

    actions = [6] * num_agents
    # first determine if an agent is working on a high priority task, if so then all other agents
    # are only allowed to work on high priority tasks
    for agent in range(num_agents):
        if env.agent_task_status[agent] == env.AgentWorkingStatus.NOT_WORKING:
            task, priority = base_exec.get_next_task(agent)
            env.agent_performing_task[agent] = task
            env.agent_working_priority[agent] = priority
            env.states[agent] = (env.states[agent][0], 0, None)
            if task is not None:
                if priority > 0:
                    print(env.agent_working_priority)
                    print(f"priority task: {task} allocated to agent: {agent}")
                    for agent_ in range(num_agents):
                        try:
                            env.warehouse_api.update_rack(
                                env.current_rack_positions[env.agent_performing_task[agent_]], 
                                env.orig_rack_positions[env.agent_performing_task[agent_]]
                            )
                        except Exception as e:
                            continue
                env.agent_task_status[agent] = env.AgentWorkingStatus.WORKING
        else:
            # Check if the agent's task has been completed
            if env.agent_performing_task[agent] is not None:
                status = base_exec.check_done(env.agent_performing_task[agent])
                #print("task: ", env.agent_performing_task[agent], "status: (", status, task_progress[status], ")")
                if task_progress[status] in ["success", "fail"]:
                    print(f"Task {env.agent_performing_task[agent]} -> {task_progress[status]}")
                    # Publish task failure to redis
                    if env.agent_working_priority[agent] == 1:
                        env.agent_working_priority[agent] = 0
                    if task_progress[status] == "fail":
                        # if the task has failed but it was originally a failed task then 
                        # we need to not overwrite the original rack position 
                        if env.agent_performing_task[agent] in env.failure_task_progress.keys():
                            rack_start = env.failure_task_progress[env.agent_performing_task[agent]]["original_position"]
                        else:
                            rack_start = env.orig_rack_positions[env.agent_performing_task[agent]]
                        
                        if rack_start != env.current_rack_positions[env.agent_performing_task[agent]]:
                            r.publish('executor-channel', json.dumps({
                                'event_type': "task_failure",
                                'rack_start': rack_start,
                                'feed': env.warehouse_api.task_feeds[env.agent_performing_task[agent]],
                                'current_rack_position': env.current_rack_positions[env.agent_performing_task[agent]],
                                'task_number': env.agent_performing_task[agent]
                            }))
                        else:
                            env.current_rack_positions[env.agent_performing_task[agent]] = None
                            env.orig_rack_positions[env.agent_performing_task[agent]]
                    else:
                        if env.agent_performing_task[agent] in env.failure_task_progress.keys():
                            cached = env.failure_task_progress[env.agent_performing_task[agent]]
                            print(f"updated rack position from {cached['start_position']}" 
                                  f"-> {cached['original_position']}")
                            r.publish('executor-channel', json.dumps({
                                'event_type': 'rack_update',
                                'original_position': cached['original_position'],
                                'start_position': cached['start_position']
                            }))
                            env.warehouse_api.update_rack(
                                cached['original_position'],
                                cached['start_position']
                            )
                            env.failure_task_progress.pop(env.agent_performing_task[agent], None)
                        if not env.failure_task_progress:
                            env.warehouse_api.set_psuccess(0.99)

                    env.agent_task_status[agent] = env.AgentWorkingStatus.NOT_WORKING
                    env.agent_rack_positions[agent] = None
                    # todo add cleanup method here to remove the completed tasks from memory

        # With the current task check what the dfa state is
        if env.agent_performing_task[agent] is not None:
            q = base_exec.dfa_current_state(env.agent_performing_task[agent])
            env.warehouse_api.set_task_(env.agent_performing_task[agent])
            try:
                actions[agent] = base_exec.get_action(agent, env.agent_performing_task[agent], env.states[agent], q)
            except:
                print(env.warehouse_api.racks)

    # step the agent forward one timestep
    if 1 in env.agent_working_priority:
        env.warehouse_api.set_psuccess(1.0)
        actions = [a if env.agent_working_priority[i] == 1 else 6 for (i, a) in enumerate(actions)]
        # update the environment with the rack positions
        #print(actions)
    if any(i for i in actions if i != 6):
        #print("rendering frame")
        env.renderer.render_step()
        frames.append(env.render())
    obs, rewards, dones, info = env.step(actions)

    # Step the DFA forward
    for agent in range(num_agents):
        current_task = env.agent_performing_task[agent]
        if current_task is not None:
            q = base_exec.dfa_current_state(current_task)
            base_exec.dfa_next_state(current_task, q, info[agent]["word"])
            qprime = base_exec.dfa_current_state(current_task)

save_frames_as_gif(frames)
