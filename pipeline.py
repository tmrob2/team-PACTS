import redis
import time
import ce
import argparse

# create a subscriber to listen to messages
r = redis.Redis(host='localhost', port=6379, db=0)

p = r.pubsub()
p.subscribe('dfa-channel')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Task processing channel')
    parser.add_argument('--interval', dest='interval', 
        help='the interval of a batches before an SCPM is created and processed')
    args = parser.parse_args()
    if not args.interval:
        parser.error("interval must be included")

    # A multiagent team to conduct a set of multiobjective missions
    team = ce.Team()
    mission = ce.Mission() 

    batch_count = args.interval

    while True:
        # get i interval messages from the channel before constructing an SCPM
        message = p.get_message()
        # we need to collect a batch either by time interval or fixed count of 
        # messages
        if message:
            data = message["data"].decode('UTF-8')
            # message is a string and we want to try and convert it to a task
            try:
                if data['type'] == "task":
                    task = ce.task_from_str()
                    print("Received Task")
                    mission.add_task(task)
                elif data['type'] == "agent":
                    agent = ce.agent_from_str()
                    team.add_agent(agent) 
                    print("Received Agent. Team size:", team.size)
                    # The messages to form agents will only come through at the 
                    # beginning of the stream loop
                    batch_count += 1
            except:
                print(f"Could not convert msg: {message} to task")
    
        # construct and SCPM to process the tasks
        scpm = ce.SCPM(team, mission)