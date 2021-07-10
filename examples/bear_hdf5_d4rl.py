import sys
def my_except_hook(exctype, value, traceback):
    import urllib
    if exctype == RuntimeError:
        print("-"*10 + "RuntimeError Encountered"+ "-"*10)
        print(exctype, value, traceback)
        import os
        os._exit(99)
    elif exctype == urllib.error.URLError:
        print("-"*10 + "URLError Encountered"+ "-"*10)
        print(exctype, value, traceback)
        import os
        os._exit(99)
    elif exctype == urllib.error.HTTPError:
        print("-"*10 + "HTTPError Encountered"+ "-"*10)
        print(exctype, value, traceback)
        import os
        os._exit(99)
    else:
        sys.__excepthook__(exctype, value, traceback)
sys.excepthook = my_except_hook

from gym.envs.mujoco import HalfCheetahEnv, HopperEnv, AntEnv, Walker2dEnv

import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector, CustomMDPPathCollector
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic, VAEPolicy
from rlkit.torch.sac.bear import BEARTrainer
from rlkit.torch.networks import FlattenMlp, FlattenDropout_Mlp, FlattenEnsemble_Mlp
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
import numpy as np

import h5py, argparse, os
import gym
import d4rl

def load_hdf5(dataset, replay_buffer, max_size):
    all_obs = dataset['observations']
    all_act = dataset['actions']
    N = min(all_obs.shape[0], max_size)

    _obs = all_obs[:N-1]
    _actions = all_act[:N-1]
    _next_obs = all_obs[1:]
    _rew = np.squeeze(dataset['rewards'][:N-1])
    _rew = np.expand_dims(np.squeeze(_rew), axis=-1)
    _done = np.squeeze(dataset['terminals'][:N-1])
    _done = (np.expand_dims(np.squeeze(_done), axis=-1)).astype(np.int32)

    max_length = 1000
    ctr = 0
    ## Only for MuJoCo environments
    ## Handle the condition when terminal is not True and trajectory ends due to a timeout
    for idx in range(_obs.shape[0]):
        if ctr  >= max_length - 1:
            ctr = 0
        else:
            replay_buffer.add_sample_only(_obs[idx], _actions[idx], _rew[idx], _next_obs[idx], _done[idx])
            ctr += 1
            if _done[idx][0]:
                ctr = 0
    ###

    print (replay_buffer._size, replay_buffer._terminals.shape)


def experiment(variant):
    eval_env = gym.make(variant['env_name'])
    expl_env = eval_env

    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size

    M = variant['layer_size']
    if variant['algorithm']=='BEAR':
        qf1 = FlattenMlp(
            input_size=obs_dim + action_dim,
            output_size=1,
            hidden_sizes=[M, M,],
        )
        qf2 = FlattenMlp(
            input_size=obs_dim + action_dim,
            output_size=1,
            hidden_sizes=[M, M,],
        )
        target_qf1 = FlattenMlp(
            input_size=obs_dim + action_dim,
            output_size=1,
            hidden_sizes=[M, M,],
        )
        target_qf2 = FlattenMlp(
            input_size=obs_dim + action_dim,
            output_size=1,
            hidden_sizes=[M, M,],
        )
    else:
        raise NameError("algorithm undefined: {}".format(variant['algorithm']))
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M,], 
    )
    vae_policy = VAEPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[750, 750],
        latent_dim=action_dim * 2,
    )
    eval_path_collector = CustomMDPPathCollector(
        eval_env,
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        policy,
    )
    buffer_filename = None
    if variant['buffer_filename'] is not None:
        buffer_filename = variant['buffer_filename']
    
    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )
    load_hdf5(eval_env.unwrapped.get_dataset(), replay_buffer, max_size=variant['replay_buffer_size'])
    
    if variant['algorithm']=='BEAR':
        trainer = BEARTrainer(
            env=eval_env,
            policy=policy,
            qf1=qf1,
            qf2=qf2,
            target_qf1=target_qf1,
            target_qf2=target_qf2,
            vae=vae_policy,
            **variant['trainer_kwargs']
        )
    else:
        raise NameError("algorithm undefined: {}".format(variant['algorithm']))
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        batch_rl=True,
        q_learning_alg=True,
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()

if __name__ == "__main__":
    # noinspection PyTypeChecker
    parser = argparse.ArgumentParser(description='BEAR-runs')
    parser.add_argument("--env", type=str, default='halfcheetah-medium-v0')
    parser.add_argument("--algo_name", type=str, default='BEAR')
    parser.add_argument("--gpu", default='0', type=str)
    parser.add_argument('--qf_lr', default=3e-4, type=float)
    parser.add_argument('--policy_lr', default=1e-4, type=float)
    parser.add_argument('--mmd_sigma', default=20, type=float)
    parser.add_argument('--kernel_type', default='gaussian', type=str)
    parser.add_argument('--target_mmd_thresh', default=0.07, type=float)
    parser.add_argument('--num_samples', default=100, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--drop_rate', default=0.1, type=float)
    parser.add_argument('--all_saves', default="saves", type=str)
    parser.add_argument('--trial_name', default="", type=str)
    parser.add_argument('--bolt', default=False, type=bool)
    parser.add_argument('--nepochs', default=3000, type=int)
    parser.add_argument('--log_dir', default='./default/', type=str)    # Logging directory
    parser.add_argument('--std_lambda', default=0.5, type=float)
    parser.add_argument('--discount_lambda', default=0.5, type=float)
    parser.add_argument('--clip_bottom', default=0.0, type=float)
    parser.add_argument('--clip_top', default=1.0, type=float)
    parser.add_argument('--use_exp_weight', default=True, type=bool)
    parser.add_argument('--var_Pi', default=False, type=bool)
    parser.add_argument('--q_penalty', default=0.0, type=float)
    parser.add_argument('--use_exp_penalty', default=False, type=bool)
    parser.add_argument('--var_discount', default=False, type=bool)
    parser.add_argument('--SN', default=False, type=bool)
    args = parser.parse_args()
    if args.bolt:
        import turibolt as bolt
        args.all_saves = bolt.ARTIFACT_DIR
    if not os.path.exists(os.path.join(args.all_saves,"results")):
        os.makedirs(os.path.join(args.all_saves,"results"))
    
    variant = dict(
        algorithm=args.algo_name,
        version="normal",
        layer_size=256,
        replay_buffer_size=int(2E6),
        buffer_filename=None, #halfcheetah_101000.pkl',
        load_buffer=True,
        env_name=args.env,
        drop_rate=args.drop_rate,
        spectral_norm=args.SN,
        algorithm_kwargs=dict(
            num_epochs=args.nepochs,
            num_eval_steps_per_epoch=5000,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=1000,
            max_path_length=1000,
            batch_size=256,
            num_actions_sample=args.num_samples,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=args.policy_lr,
            qf_lr=args.qf_lr,
            reward_scale=1,

            # BEAR specific params
            mode='auto',
            kernel_choice=args.kernel_type,
            policy_update_style='0',
            mmd_sigma=args.mmd_sigma,
            target_mmd_thresh=args.target_mmd_thresh,

            #dropout specific params
            std_lambda=args.std_lambda,
            discount_lambda=args.discount_lambda,
            clip_bottom=args.clip_bottom,
            clip_top=args.clip_top,
            use_exp_weight=args.use_exp_weight,
            var_discount=args.var_discount,
            var_Pi=args.var_Pi,
            q_penalty=args.q_penalty,
            use_exp_penalty=args.use_exp_penalty,
        ),
    )
    file_name = args.trial_name
    rand = np.random.randint(0, 100000)
    # setup_logger(os.path.join('BEAR_launch', str(rand)), variant=variant, base_log_dir='./data')
    setup_logger(file_name, variant=variant, base_log_dir=args.all_saves, name = file_name, log_dir=os.path.join(args.all_saves,args.log_dir,file_name), bolt = args.bolt)
    ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    experiment(variant)
