# CSCE-642: Deep Reinforcement Learning

## Setup

**Python 3.11 or newer.** Gymnasium 1.2 dropped support for Python 3.9, so the
older 3.9.16 environment used in previous offerings no longer works.

SWIG is required for installing Box2D (used by the LunarLander domains). Install
it on Linux with

```bash
sudo apt-get install swig build-essential python3-dev
```

on Mac with

```bash
brew install swig
```

or on Windows by following the instructions
[here](https://open-box.readthedocs.io/en/latest/installation/install_swig.html).

MuJoCo (used by the Hopper and HalfCheetah domains) ships prebuilt wheels and
needs no extra system packages.

We recommend conda + pip or venv + pip:

```bash
conda create -n csce642 python=3.11
conda activate csce642
pip install -r requirements.txt
```

or

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Verify the install:

```bash
python run.py -s random -d Gridworld -e 2 --no-plots
python run.py -s random -d LunarLander-v3 -e 2 -t 50 --no-plots
```

## Running

```bash
python run.py -s <solver> -d <domain> [options]
python autograder.py <solver>
```

Run `python run.py -h` for the full option list.

## A note on epsilon-greedy policies

Throughout this codebase a stochastic policy returns `pi(.|s)` as a **vector of
action probabilities**, one entry per action, not a sampled action. This applies
to `make_epsilon_greedy_policy` (Monte Carlo) and to `epsilon_greedy`
(Q-Learning, SARSA, approximate Q-Learning, DQN).

When you need to act on that vector, use the provided helper:

```python
probs = self.epsilon_greedy(state)
action = self.sample(probs)
```

## Domain versions

Gymnasium renamed several environments in the 1.0 release. This codebase uses
the current ids:

| Domain | Id |
| --- | --- |
| Cart Pole | `CartPole-v1` |
| Mountain Car | `MountainCar-v0` |
| Frozen Lake | `FrozenLake-v1` |
| Lunar Lander | `LunarLander-v3` |
| Lunar Lander (continuous) | `LunarLanderContinuous-v3` |
| Hopper | `Hopper-v5` |
| Half Cheetah | `HalfCheetah-v5` |

`Gridworld`, `Blackjack`, `CliffWalking`, and `WindyGridworld` are provided
locally under `lib/envs/` and are not Gymnasium-registered environments.
