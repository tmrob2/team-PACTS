import redis
import warehouse 
from warehouse.envs.warehouse import Warehouse
import ce
import argparse
import gym
import itertools
import json

# create a subscriber to listen to messages
r = redis.Redis(host='localhost', port=6379, db=0)

p = r.pubsub()
p.subscribe('dfa-channel')
p.subscribe('executor-channel')
thread = None

NUM_AGENTS = 3

# ------------------------------------------------------------------------------
# SETUP: Construct the structures for agent to recognise task progress
# ------------------------------------------------------------------------------

task_progress = {0: "initial", 1: "in_progress", 2: "success", 3: "fail"}

# Set the initial agent locations up front
# We can set the feed points up front as well because they are static
init_agent_positions = [(0, 0), (4, 0), (0, 4)]
size = 10

# ------------------------------------------------------------------------------
# Env Setup: Construct the warehouse model as an Open AI gym python environment
# ------------------------------------------------------------------------------
env: Warehouse = gym.make(
    "Warehouse-v0", 
    initial_agent_loc=init_agent_positions, 
    nagents=NUM_AGENTS,
    feedpoints=[],
    render_mode=None,
    size=size,
    seed=4321,
    prob=0.99,
    disable_env_checker=True,
)
# We have to set the tasks racks and task feeds which depend on the number of tasks

print("random task racks: ", env.warehouse_api.task_racks_start)

obs = env.reset()
print("Initial observations: ", obs)
print("Agent rack positions: ", env.agent_rack_positions)

# ------------------------------------------------------------------------------
# Tasks: Construct a DFA transition function and build the Mission from this
# ------------------------------------------------------------------------------

def warehouse_replenishment_task():
    task = ce.DFA(list(range(0, 8)), 0, [5], [7], [6])
    # attempt to goto the rack positon without carrying anything
    omega = set(env.warehouse_api.words)

    # The first transition determines if the label is at the rack
    task.add_transition(0, "RS_NC", 1)
    excluded_words = ['_'.join(x) for x in list(itertools.product(["RS", "RE", "NFR", "F"], ["P", "D", "CR", "CNR"]))]
    for w in excluded_words: 
        task.add_transition(0, f"{w}", 7)
    excluded_words.append("RS_NC")
    for w in omega.difference(set(excluded_words)):
        task.add_transition(0, f"{w}", 0)
    # The second transition determines whether the agent picked up the rack at the 
    # required coord
    task.add_transition(1, "RS_P", 2)
    excluded_words = ['_'.join(x) for x in list(itertools.product(["NFR"], ["P"]))]
    for w in excluded_words:
        task.add_transition(1, f"{w}", 7)
    excluded_words.append("RS_P")
    for w in omega.difference(set(excluded_words)):
        task.add_transition(1, f"{w}", 1)
    # The third transition takes the agent to the feed position while carrying
    task.add_transition(2, "F_CNR", 3)
    excluded_words = ['_'.join(x) for x in list(itertools.product(["F", "RS", "RE", "NFR"], ["NC", "P", "D", "CR"]))]
    for w in excluded_words:
        task.add_transition(2, f"{w}", 7)
    excluded_words.append("F_CNR")
    for w in omega.difference(set(excluded_words)):
        task.add_transition(2, f"{w}", 2)
    # The fourth transition takes the agent from the feed position while carrying 
    # back to the rack position
    task.add_transition(3, "RS_CNR", 4)
    excluded_words = ['_'.join(x) for x in list(itertools.product(["F", "RS", "RE", "NFR"], ["NC", "P", "D", "CR"]))]
    #excluded_words.append("RS_CNR")
    for w in excluded_words:
        task.add_transition(3, f"{w}", 7)
    excluded_words.append("RS_CNR")
    for w in omega.difference(set(excluded_words)):
        task.add_transition(3, f"{w}", 3)
    # The fifth transition tells the agent to drop the rack at the required square
    task.add_transition(4, "RS_D", 5)
    for w in omega.difference(set(["RS_D"])):
        task.add_transition(4, f"{w}", 4)
    for w in omega:
        task.add_transition(5, f"{w}", 6)
    for w in omega:
        task.add_transition(6, f"{w}", 6)
    for w in omega:
        task.add_transition(7, f"{w}", 7)
    
    return task

def warehouse_retry_task():
    task = ce.DFA(list(range(0, 8)), 0, [5], [7], [6])
    # attempt to goto the rack positon without carrying anything
    omega = set(env.warehouse_api.words)

    # The first transition determines if the label is at the rack
    task.add_transition(0, "RS_NC", 1)
    excluded_words = ['_'.join(x) for x in list(itertools.product(["RS", "RE", "NFR", "F"], ["P", "D", "CR", "CNR"]))]
    excluded_words.append("RE_NC")
    for w in excluded_words: 
        task.add_transition(0, f"{w}", 7)
    excluded_words.append("RS_NC")
    for w in omega.difference(set(excluded_words)):
        task.add_transition(0, f"{w}", 0)
    # The second transition determines whether the agent picked up the rack at the 
    # required coord
    task.add_transition(1, "RS_P", 2)
    excluded_words = ['_'.join(x) for x in list(itertools.product(["NFR", "RE"], ["P"]))]
    for w in excluded_words:
        task.add_transition(1, f"{w}", 7)
    excluded_words.append("RS_P")
    for w in omega.difference(set(excluded_words)):
        task.add_transition(1, f"{w}", 1)
    # The third transition takes the agent to the feed position while carrying
    task.add_transition(2, "F_CNR", 3)
    excluded_words = ['_'.join(x) for x in list(itertools.product(["F", "RS", "RE", "NFR"], ["NC", "P", "D", "CR"]))]
    for w in excluded_words:
        task.add_transition(2, f"{w}", 7)
    excluded_words.append("F_CNR")
    for w in omega.difference(set(excluded_words)):
        task.add_transition(2, f"{w}", 2)
    # The fourth transition takes the agent from the feed position while carrying 
    # back to the rack position
    task.add_transition(3, "RE_CNR", 4)
    excluded_words = ['_'.join(x) for x in list(itertools.product(["F", "RS", "RE", "NFR"], ["NC", "P", "D", "CR"]))]
    excluded_words.append("RS_CNR")
    for w in excluded_words:
        task.add_transition(3, f"{w}", 7)
    excluded_words.append("RE_CNR")
    for w in omega.difference(set(excluded_words)):
        task.add_transition(3, f"{w}", 3)
    # The fifth transition tells the agent to drop the rack at the required square
    task.add_transition(4, "RE_D", 5)
    for w in omega.difference(set(["RE_D"])):
        task.add_transition(4, f"{w}", 4)
    for w in omega:
        task.add_transition(5, f"{w}", 6)
    for w in omega:
        task.add_transition(6, f"{w}", 6)
    for w in omega:
        task.add_transition(7, f"{w}", 7)
    
    return task

task_progress = {0: "initial", 1: "in_progress", 2: "success", 3: "fail"}

dfa1 = warehouse_replenishment_task()
dfa2 = warehouse_retry_task()

# ------------------------------------------------------------------------------
# Executor: Define a new executor which will be used to continually run agents
# ------------------------------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Task processing channel')
    parser.add_argument('--interval', dest='interval', 
        help='the interval of a batches before an SCPM is created and processed')
    parser.add_argument('--eps', dest="eps", help="Epsilon value for value iteration")
    args = parser.parse_args()
    if not args.interval:
        parser.error("interval must be included")

    # A multiagent team to conduct a set of multiobjective missions
    # constuct the environment
    
    
    eps = 0.0001 if not args.eps else float(args.eps) # set the epsilon value for value iteration

    batch_count = 0
    force_batch = False
    task_map = {}
    mission = ce.Mission()
    data_decoded = False
    while True:
        # get i interval messages from the channel before constructing an SCPM
        message = p.get_message() # this is a redis framework method
        # we need to collect a batch either by time interval or fixed count of 
        # messages
        if message:
            # message is a string and we want to try and convert it to a task
            #try:
            if message['channel'] == b'executor-channel' \
                and not isinstance(message['data'], int):
                data_ = message['data'].decode('utf-8')
                data = json.loads(data_)
                if data['event_type'] == 'task_failure':
                    rack_start = data['current_rack_position']
                    rack_end = data['rack_start']
                    number = data['task_number']
                    # update the rack position in the warehouse for the next batch
                    env.warehouse_api.add_task_rack_start(batch_count, tuple(rack_start))
                    env.warehouse_api.add_task_feed(batch_count, tuple(data['feed']))
                    env.warehouse_api.add_task_rack_end(batch_count, tuple(rack_end))
                    env.warehouse_api.update_rack(tuple(rack_start), tuple(rack_end))
                    mission.add_task(dfa2)
                    task_map[batch_count] = (number, 1)
                    print(f"Task failure: start {rack_start}, end: {rack_end}, feed: {data['feed']}")
                    batch_count += 1
                elif data['event_type'] == 'rack_update':
                    print(f"updated rack position from {tuple(data['start_position'])}" 
                          f"-> {tuple(data['original_position'])}")
                    env.warehouse_api.update_rack(
                        tuple(data['original_position']), 
                        tuple(data['start_position'])
                    )
            elif message['channel'] == b'dfa-channel' \
                and not isinstance(message['data'], int):
                data_ = message["data"].decode('utf-8')
                data = json.loads(data_)
                if data['event_type'] == "task":
                    print("Received Task Event", data)
                    mission.add_task(dfa1)
                    task_map[batch_count] = (data['number'], 0)
                    env.warehouse_api.add_task_rack_start(batch_count, tuple(data['rack_start']))
                    env.warehouse_api.add_task_feed(batch_count, tuple(data['feed']))
                    batch_count += 1
                elif data['event_type'] == 'end_stream':
                    # send a poison pill to the thread
                    print("received tear down event, ending listener")
                    break
                elif data['event_type'] == 'force_batch':
                    print("received force event")
                    force_batch = True
                # construct and SCPM to process the tasks

            if batch_count == int(args.interval) or force_batch:
                if force_batch: 
                    print("start", env.warehouse_api.task_racks_start)
                    print("end", env.warehouse_api.task_racks_end)
                    print("feed", env.warehouse_api.task_feeds)
                    print("mission", mission.size)
                    
                print("batch count", batch_count)
                executor = ce.SerialisableExecutor(NUM_AGENTS)
                w = [0] * NUM_AGENTS + [1./ batch_count] * batch_count
                # TODO Target needs to be dynamic 
                target = [-65., -65., 90.] + [0.7] * batch_count
                executor.insert_task_map(task_map)
                scpm = ce.SCPM(mission, NUM_AGENTS, list(range(6)))
                sol_not_found = True
                while sol_not_found:
                    try:
                        tnew = ce.scheduler_synth(
                            scpm, env.warehouse_api, w, target, eps, executor
                        )
                        sol_not_found = False
                    except:
                        continue
                # send the executor over redis
                r.publish('executor-channel', json.dumps(
                    {
                        "event_type": "execute", 
                        "executor": executor.to_json(),
                        "task-feed": env.warehouse_api.task_feeds,
                        "task-rack-start": env.warehouse_api.task_racks_start,
                        "task-rack-end": env.warehouse_api.task_racks_end,
                        "batch-size": batch_count
                    }))
                # clean up 
                env.warehouse_api.clear_env()
                mission = ce.Mission()
                batch_count = 0
                task_map = {}
                if force_batch: force_batch = False
                