
from threading import Thread
import threading
import redis
import random
import json
import warehouse 
from warehouse.envs.warehouse import Warehouse
import gym
import pygame

pygame.init()


class EventThread(Thread):

    class TaskEvent:
        def __init__(
                self, 
                rack_start, 
                rack_end, 
                feed, number, 
                task_type=None, 
                priority=None
            ):
            self.event_type = "task"
            self.rack_start = rack_start
            self.feed = feed
            self.rack_end = rack_end
            self.number = number
            self.task_type = task_type
            self.priority = priority

    def __init__(
            self, 
            queue, 
            nagents, 
            init_agent_positions,
            feedpoints,
            size,
            args=(), 
            kwargs=()
        ):
        threading.Thread.__init__(self, args=(), kwargs=None)
        self.daemon = True
        self.queue = queue
        self.task_counter = 0
        self.failure_queue = []

        self.r = redis.Redis(host='localhost', port=6379, db=0)
        self.p = self.r.pubsub()
        self.p.subscribe('dfa-channel')
        self.p.subscribe('executor-channel') # need to listen for event failures

        self.FRAMES_PER_SECOND = 20
        self.milli_seconds_since_event = 0
        self.milli_seconds_until_event = random.randint(500, 2000)
        self.clock = pygame.time.Clock()

        # define the warehouse here for referencing attributes
        self.env: Warehouse = gym.make(
            "Warehouse-v0", 
            initial_agent_loc=init_agent_positions, 
            nagents=nagents,
            feedpoints=feedpoints,
            render_mode=None,
            size=size,
            seed=4321,
            prob=0.99,
            disable_env_checker=True,
        )

    def run(self):
        print("running thread")
        while True:
            val = self.queue.get()
            if val['event'] == 'send one random task':
                self.one_task_event()
            elif val['event'] == 'send a batch of tasks':
                self.send_k_tasks(val['k'])
            elif val['event'] == 'task stream':
                self.continuous()
            elif val['event'] == "process failure":
                self.process_failed_tasks()
            elif val['event'] == 'stop task stream' or \
                     val['event'] == 'tear down listener':
                self.send_poison_pill()
                break
            # continuously send messages (tasks) to the dfa subscriber

    #
    # Send Events to the listener
    #
    def one_task_event(self):
        rack = random.sample([*self.env.warehouse_api.racks], 1)
        feed = random.sample(self.env.warehouse_api.feedpoints, 1)
        new_task = self.TaskEvent(rack, rack, feed, self.task_counter)
        msg = json.dumps(new_task.__dict__)
        self.r.publish('dfa-channel', msg)
        self.task_counter += 1

    def send_k_tasks(self, k):
        # randomly select k warehouse rack positions without replacement
        racks_sample = random.sample([*self.env.warehouse_api.racks], k)
        feeds = random.choices(self.env.warehouse_api.feedpoints, k=k)
        for _ in range(k):
            rack = racks_sample.pop()
            new_task = self.TaskEvent(
                rack, None, feeds.pop(), self.task_counter
            )
            self.r.publish('dfa-channel', json.dumps(new_task.__dict__))
            self.task_counter += 1
        self.task_counter

    def process_failed_tasks(self):
        self.r.publish('dfa-channel', json.dumps({"event_type": "force_batch"}))

    def send_poison_pill(self):
        d = {"event_type": "end_stream"}
        self.r.publish("dfa-channel", json.dumps(d))

    def continuous(self):
        milliseconds_since_event += self.clock.tick(self.FRAMES_PER_SECOND)

        # handle input here

        if milliseconds_since_event > milliseconds_until_event:
            # perform your random event
            print("\rsend a new continuous event")
            milliseconds_until_event = random.randint(500, 2001)
            milliseconds_since_event = 0
                
