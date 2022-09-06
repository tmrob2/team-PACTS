import warehouse
#import logging
import gym

#logging.getLogger().setLevel(logging.CRITICAL)


# Set the initial agent locations up front
# We can set the feed points up front as well because they are static
init_agent_positions = [(0, 0), (9, 9)]
xsize, ysize = 10, 10
feedpoints = [(xsize - 1, ysize // 2)]
env = gym.make(
    "Warehouse-v0", 
    initial_agent_loc=init_agent_positions, 
    nagents=2,
    feedpoints=feedpoints
)

obs = env.reset()
print("Initial observations: ", obs)
print("Agent rack positions: ", env.agent_rack_position)
