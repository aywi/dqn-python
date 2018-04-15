# dqn-python

Deep Q-Networks in Python

In Conda environment, please replace `python3` as `python`.

## 1. Linear Q-Networks

For CartPole-v0:

```
python3 ./DQN_Implementation.py --network_type LQN --env CartPole-v0 --gamma 0.99 --train_episode 200

Results:
Training Episode:  200 Iteration:   23084 Total Reward:  199.0
Test Average Total Reward: 198.9±4.7
```

For MountainCar-v0:

```
python3 ./DQN_Implementation.py --network_type LQN --env MountainCar-v0 --gamma 1 --train_episode 3000

Results:
Training Episode: 3000 Iteration:  600000 Total Reward: -200.0
Test Average Total Reward: -200.0±0.0
```

## 2. Linear Q-Networks with Experience Replay

For CartPole-v0:

```
python3 ./DQN_Implementation.py --network_type Replay_LQN --env CartPole-v0 --gamma 0.99 --train_episode 500

Results:
Training Episode:  500 Iteration:   71018 Total Reward:  200.0
Test Average Total Reward: 200.0±0.0
```

For MountainCar-v0:

```
python3 ./DQN_Implementation.py --network_type Replay_LQN --env MountainCar-v0 --gamma 1 --train_episode 3000

Results:
Training Episode: 3000 Iteration:  600000 Total Reward: -200.0
Test Average Total Reward: -200.0±0.0
```

## 3. Deep Q-Networks

For CartPole-v0:

```
python3 ./DQN_Implementation.py --network_type DQN --env CartPole-v0 --gamma 0.99 --train_episode 400

Results:
Training Episode:  400 Iteration:   47288 Total Reward:  200.0
Test Average Total Reward: 199.3±3.3
```

For MountainCar-v0:

```
python3 ./DQN_Implementation.py --network_type DQN --env MountainCar-v0 --gamma 1 --train_episode 1000

Results:
Training Episode: 1000 Iteration:  181383 Total Reward:  -89.0
Test Average Total Reward: -133.8±30.5
```

## 4. Dueling Deep Q-Networks

For CartPole-v0:

```
python3 ./DQN_Implementation.py --network_type Dueling_DQN --env CartPole-v0 --gamma 0.99 --train_episode 500

Results:
Training Episode:  500 Iteration:   44419 Total Reward:  200.0
Test Average Total Reward: 200.0±0.0
```

For MountainCar-v0:

```
python3 ./DQN_Implementation.py --network_type Dueling_DQN --env MountainCar-v0 --gamma 1 --train_episode 3000

Results:
Training Episode: 3000 Iteration:  454477 Total Reward: -113.0
Test Average Total Reward: -121.2±18.8
```

## 5. Convolutional Deep Q-Networks

For SpaceInvaders-v0:

```
python3 ./DQN_Implementation.py --network_type Conv_DQN --env SpaceInvaders-v0 --gamma 0.99 --train_episode 100 --memory_size 1000000 --burn_in 50000 --target_update_frequency 10000

Results:
Training Episode: 1000 Iteration:  761332 Total Reward:  105.0
Test Average Total Reward: 133.3±104.7
```
