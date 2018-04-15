# dqn-python

Deep Q-Networks in Python

In Conda environment, please replace `python3` as `python`.

## 1. Deep Q-Networks

For CartPole-v0:

```
python3 ./dqn.py --network_type DQN --env CartPole-v0 --gamma 0.99 --train_episode 400
```

For MountainCar-v0:

```
python3 ./dqn.py --network_type DQN --env MountainCar-v0 --gamma 1 --train_episode 1000
```

## 2. Dueling Deep Q-Networks

For CartPole-v0:

```
python3 ./dqn.py --network_type Dueling_DQN --env CartPole-v0 --gamma 0.99 --train_episode 500
```

For MountainCar-v0:

```
python3 ./dqn.py --network_type Dueling_DQN --env MountainCar-v0 --gamma 1 --train_episode 3000
```

## 3. Convolutional Deep Q-Networks

For SpaceInvaders-v0:

```
python3 ./dqn.py --network_type Conv_DQN --env SpaceInvaders-v0 --gamma 0.99 --train_episode 100 --memory_size 1000000 --burn_in 50000 --target_update_frequency 10000
```
