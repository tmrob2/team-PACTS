from email import utils
from queue import Queue
import random
import redis
import json
import inquirer
from utils.event_thread import EventThread


#
# Params
#
NUM_AGENTS = 3
init_agent_positions = [(0, 0), (4, 0), (0, 4)]
size = 10
feedpoints = [(size - 1, size // 2)]

class TaskOptions:
    SEND_BATCH = 'send a batch of tasks'
    SEND_ONE = 'send one random task'
    SEND_POISON = 'tear down listener'
    SEND_CONTINUOUS = 'task stream'
    STOP_CONTINUOUS = 'stop task stream'



r = redis.Redis(host='localhost', port=6379, db=0)
p = r.pubsub()
p.subscribe('dfa-channel')
p.subscribe('executor-channel') # need to listen for event failures

failed_task_queue = []

q = Queue()
event_thread = EventThread(q, NUM_AGENTS, init_agent_positions, feedpoints, size)
event_thread.start()

if __name__ == "__main__":
    task_counter = 0
    continous_tasks = False
    while True:
        message = p.get_message()
        if message:
            # message is a string and we want to try and convert it to a task
            try:
                if message['channel'] == 'executor-channel':
                    data_ = message["data"].decode('utf-8')
                    #print("data_", data_)
                    data = json.loads(data_)
                    if data['event_type'] == 'task_failure':
                        print("task failure received", data)
            except:
                print("Could not decode message")
        if not continous_tasks:
            questions = [
                inquirer.List('task',
                            message="Choose a task option",
                            choices=[
                                        TaskOptions.SEND_ONE,
                                        TaskOptions.SEND_BATCH,
                                        TaskOptions.SEND_POISON,
                                        TaskOptions.SEND_CONTINUOUS
                                    ]
                )
            ]

            answers = inquirer.prompt(questions, theme=inquirer.themes.GreenPassion())
            if answers["task"] == TaskOptions.SEND_BATCH:
                qbatch = [
                        inquirer.Text("batch", message="How many (must be int)?")
                    ]
                answers = inquirer.prompt(qbatch, theme=inquirer.themes.GreenPassion())
                k = int(answers["batch"])
                event_thread.queue.put(
                    {'event': TaskOptions.SEND_BATCH, 'k': k}
                )

            elif answers["task"] == TaskOptions.SEND_POISON:
                event_thread.queue.put({
                    'event': TaskOptions.SEND_POISON
                })
                break
            elif answers['task'] == TaskOptions.SEND_CONTINUOUS:
                continous_tasks = True
        else:
            
            questions = [
                inquirer.List('task',
                            message="Choose a task option",
                            choices=[
                                        TaskOptions.STOP_CONTINUOUS,
                                        TaskOptions.SEND_POISON,
                                    ]
                )
            ]

            answers = inquirer.prompt(questions, theme=inquirer.themes.GreenPassion())
            if answers["task"] == TaskOptions.STOP_CONTINUOUS:
                continous_tasks = False

            elif answers["task"] == TaskOptions.SEND_POISON:
                event_thread.queue.put({
                    'event': TaskOptions.SEND_POISON
                })
                break
            

    