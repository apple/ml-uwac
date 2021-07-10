# Uncertainty Weighted Actor Critic (UWAC)

This is the official implementation of Uncertainty Weighted Actor Critic (UWAC) that accompanies the research paper, Uncertainty Weighted Actor Critic for Offline Reinforcement Learning (ICML 2021).

Uncertainty Weighted Actor-Critic (UWAC), an algorithm detects OOD state-action pairs and down-weights their contribution in the training objectives accordingly. We adopt a practical and effective dropout-based uncertainty estimation method that introduces very little overhead over existing RL algorithms.

## Getting Started
This codebase is built off of the official implementation of BEAR (https://github.com/rail-berkeley/d4rl_evaluations/tree/master/bear) and rlkit (https://github.com/vitchyr/rlkit/). In order to run UWAC, follow the installation instructions for rlkit as shown below, then install D4RL(https://github.com/rail-berkeley/d4rl).

Then in order to run UWAC, an example command is:
```
python examples/UWAC_hdf5_d4rl.py --env='halfcheetah-medium-v0' --policy_lr=1e-4 --num_samples=100
```
where, `env` refers to a d4rl environment, `policy_lr` is the policy learning rate.

To run BEAR (baseline), an example command is:
```
python examples/bear_hdf5_d4rl.py --env='halfcheetah-medium-v0' --policy_lr=1e-4 --num_samples=100
```

See readme from RLkit (https://github.com/vitchyr/rlkit) for troubleshooting with respect to RLkit problems.

# License
This code is released under the [LICENSE](LICENSE) terms.