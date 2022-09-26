from email import utils
from queue import Queue
import random
import redis
import json
import inquirer
from utils.event_thread import EventThread
import pygame

pygame.init()

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
    FAILURE = "process failure"

r = redis.Redis(host='localhost', port=6379, db=0)
p = r.pubsub()
p.subscribe('dfa-channel')
p.subscribe('executor-channel') # need to listen for event failures

failed_task_queue = []

q = Queue()
event_thread = EventThread(q, NUM_AGENTS, init_agent_positions, feedpoints, size)
event_thread.start()

FRAMES_PER_SECOND = 20
milliseconds_since_event = 0
milliseconds_until_event = random.randint(500, 2000)
clock = pygame.time.Clock()
state = 0


if __name__ == "__main__":
    task_counter = 0
    continous_tasks = False
    while True:
        if not continous_tasks:
            questions = [
                inquirer.List('task',
                            message="Choose a task option",
                            choices=[
                                        TaskOptions.SEND_ONE,
                                        TaskOptions.SEND_BATCH,
                                        TaskOptions.SEND_POISON,
                                        TaskOptions.SEND_CONTINUOUS,
                                        TaskOptions.FAILURE
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
            elif answers['task'] == TaskOptions.SEND_CONTINUOUS:
                continous_tasks = True
            elif answers['task'] == TaskOptions.FAILURE:
                event_thread.queue.put(
                    {'event': TaskOptions.FAILURE}
                )
        else:
            milliseconds_since_event += clock.tick(FRAMES_PER_SECOND)

            # handle input here

            if milliseconds_since_event > milliseconds_until_event:
                # perform random event
                if state == 0:
                    event_thread.queue.put(
                        {'event': TaskOptions.SEND_BATCH, 'k': 10}
                    )
                    state += 1
                else:
                    event_thread.queue.put(
                        {'event': TaskOptions.FAILURE}
                    )
                    state = 0
                milliseconds_until_event = random.randint(5000, 15000)
                milliseconds_since_event = 0
            #questions = [
            #    inquirer.List('task',
            #                message="Choose a task option",
            #                choices=[
            #                            TaskOptions.STOP_CONTINUOUS,
            #                            TaskOptions.SEND_POISON,
            #                        ]
            #    )
            #]
            #answers = inquirer.prompt(questions, theme=inquirer.themes.GreenPassion())
            #if answers["task"] == TaskOptions.STOP_CONTINUOUS:
            #    continous_tasks = False
            #elif answers["task"] == TaskOptions.SEND_POISON:
            #    event_thread.queue.put({
            #        'event': TaskOptions.SEND_POISON
            #    })
            #    break
            

    