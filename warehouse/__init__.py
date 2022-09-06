from gym.envs.registration import register

register(
    id='Warehouse-v0',
    entry_point='warehouse.envs:Warehouse',
    max_episode_steps=300,
)