from collections import deque

import numpy as np
import pickle
from mujoco_py import MujocoException

from baselines.her.util import convert_episode_to_batch_major, store_args

import pickle, os, gzip
import numpy as np
from pyquaternion import Quaternion
from time import gmtime, strftime


class RolloutWorker:

    @store_args
    def __init__(self, make_env, policy, dims, logger, T, rollout_batch_size=1,
                 exploit=False, use_target_net=False, compute_Q=False, noise_eps=0,
                 random_eps=0, history_len=100, render=False, **kwargs):
        """Rollout worker generates experience by interacting with one or many environments.

        Args:
            make_env (function): a factory function that creates a new instance of the environment
                when called
            policy (object): the policy that is used to act
            dims (dict of ints): the dimensions for observations (o), goals (g), and actions (u)
            logger (object): the logger that is used by the rollout worker
            rollout_batch_size (int): the number of parallel rollouts that should be used
            exploit (boolean): whether or not to exploit, i.e. to act optimally according to the
                current policy without any exploration
            use_target_net (boolean): whether or not to use the target net for rollouts
            compute_Q (boolean): whether or not to compute the Q values alongside the actions
            noise_eps (float): scale of the additive Gaussian noise
            random_eps (float): probability of selecting a completely random action
            history_len (int): length of history for statistics smoothing
            render (boolean): whether or not to render the rollouts
        """
        self.envs = [make_env() for _ in range(rollout_batch_size)]
        assert self.T > 0

        self.info_keys = [key.replace('info_', '') for key in dims.keys() if key.startswith('info_')]

        self.success_history = deque(maxlen=history_len)
        self.success_any_history = deque(maxlen=history_len)
        self.Q_history = deque(maxlen=history_len)

        self.n_episodes = 0
        self.g = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # goals
        self.initial_o = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)  # observations
        self.initial_ag = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # achieved goals
        self.reset_all_rollouts()
        self.clear_history()

    ### ndrw touch start ####################################
    def generate_cloud(self, size=11, dimension=1.0):
        # generates point cloud with center 0,0,0
        pts = []
        for i in np.linspace(-dimension, dimension, size):
            for j in np.linspace(-dimension, dimension, size):
                for k in np.linspace(-dimension, dimension, size):
                    if abs(i) == dimension or abs(j) == dimension or abs(k) == dimension:
                        pts.append(np.array([i, j, k]))
        return pts


    def transform_cloud(self, point_cloud, quaternion, cube_position=[0.0, 0.0, 0.0]):
        pts_transformed = []
        for pt in point_cloud:
            pts_transformed.append(quaternion.rotate(pt) + cube_position)
        return pts_transformed


    def find_scaling(self, cloud_transformed, voxel_range):
        max_dimension = max(list(map(lambda x: max(x), cloud_transformed)))
        # print("find_scaling: return = ", voxel_range / max_dimension)
        return voxel_range / max_dimension


    def cloud2voxel(self, cloud, voxel_range, size=16, shift=[0, 0, 0]):
        # convert to voxel grid
        size05 = (size - 1) / 2
        voxels = np.zeros((size, size, size), dtype=np.uint8)
        pos_center = (size05, size05, size05)
        for pt in cloud:
            i, j, k = np.round(pt * size05 / voxel_range + pos_center).astype('int')
            # print(pt * size05 / voxel_range + pos_center, i, j, k)  # debugging
            i += shift[0]
            j += shift[1]
            k += shift[2]
            if i < size and j < size and k < size and i >= 0 and j >= 0 and k >= 0:
                voxels[int(i)][int(j)][int(k)] = 1
            else:
                print(i, j, k)
                quit()
        return voxels
    ### ndrw touch end ########################################

    def reset_rollout(self, i):
        """Resets the `i`-th rollout environment, re-samples a new goal, and updates the `initial_o`
        and `g` arrays accordingly.
        """
        obs = self.envs[i].reset()
        self.initial_o[i] = obs['observation']
        self.initial_ag[i] = obs['achieved_goal']
        self.g[i] = obs['desired_goal']


    def reset_all_rollouts(self):
        """Resets all `rollout_batch_size` rollout workers.
        """
        for i in range(self.rollout_batch_size):
            self.reset_rollout(i)


    def generate_rollouts(self):
        """Performs `rollout_batch_size` rollouts in parallel for time horizon `T` with the current
        policy acting on it accordingly.
        """
        self.reset_all_rollouts()

        # compute observations
        o = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)  # observations
        ag = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # achieved goals
        o[:] = self.initial_o
        ag[:] = self.initial_ag

        # generate episodes
        obs, achieved_goals, acts, goals, successes = [], [], [], [], []
        info_values = [np.empty((self.T, self.rollout_batch_size, self.dims['info_' + key]), np.float32) for key in self.info_keys]
        Qs = []

        ### ndrw touch start ##################
        quaternions = []
        cloudShapes = []
        voxelShapes = []
        voxelSpaceSize = 16
        voxelRange = 2.0
        dimension = 1.0
        cloud = self.generate_cloud(size=voxelSpaceSize, dimension=dimension)
        ### ndrw touch end ##########################

        for t in range(self.T):
            policy_output = self.policy.get_actions(
                o, ag, self.g,
                compute_Q=self.compute_Q,
                noise_eps=self.noise_eps if not self.exploit else 0.,
                random_eps=self.random_eps if not self.exploit else 0.,
                use_target_net=self.use_target_net)

            if self.compute_Q:
                u, Q = policy_output
                Qs.append(Q)
            else:
                u = policy_output

            if u.ndim == 1:
                # The non-batched case should still have a reasonable shape.
                u = u.reshape(1, -1)

            o_new = np.empty((self.rollout_batch_size, self.dims['o']))
            ag_new = np.empty((self.rollout_batch_size, self.dims['g']))
            success = np.zeros(self.rollout_batch_size)
            # compute new states and observations
            for i in range(self.rollout_batch_size):
                try:
                    # We fully ignore the reward here because it will have to be re-computed
                    # for HER.
                    curr_o_new, _, _, info = self.envs[i].step(u[i])
                    if 'is_success' in info:
                        success[i] = info['is_success']
                    o_new[i] = curr_o_new['observation']
                    ag_new[i] = curr_o_new['achieved_goal']
                    for idx, key in enumerate(self.info_keys):
                        info_values[idx][t, i] = info[key]
                    if self.render:
                        self.envs[i].render()
                except MujocoException as e:
                    return self.generate_rollouts()

            if np.isnan(o_new).any():
                self.logger.warn('NaN caught during rollout generation. Trying again...')
                self.reset_all_rollouts()
                return self.generate_rollouts()

            obs.append(o.copy())
            achieved_goals.append(ag.copy())
            successes.append(success.copy())
            acts.append(u.copy())
            goals.append(self.g.copy())
            o[...] = o_new
            ag[...] = ag_new
        obs.append(o.copy())
        achieved_goals.append(ag.copy())
        self.initial_o[:] = o

        episode = dict(o=obs,
                       u=acts,
                       g=goals,
                       ag=achieved_goals)
        ### ndrw touch start ##################
        for i in range(len(obs)):
            quat = Quaternion(obs[i][0][-4:])
            if i % 25 == 0: print(strftime("%H%M%S", gmtime()), '\t', i, '\t', quat)
            cloud_transformed = self.transform_cloud(cloud.copy(), quat)
            voxels = self.cloud2voxel(cloud_transformed, voxelRange, size=voxelSpaceSize)
            voxelShapes.append(voxels)

        senses = dict(sens_proprio=[ob[0][:24] for ob in obs],
                      sens_touch=[ob[0][54:-7] for ob in obs],
                      sens_pos_quat=[ob[0][-7:] for ob in obs],
                      sens_voxel=voxelShapes
        )
        folder_md = '../../../HandManipulateBlockTouchSensors_multisensory_data/'
        if not os.path.exists(folder_md):
            os.makedirs(folder_md)
        with gzip.GzipFile(folder_md + 'multisensory_data_ep_' + strftime("%H%M%S", gmtime()) + '.pgz', 'w') as f:
            pickle.dump(senses, f)
        ### ndrw touch end ##########################

        for key, value in zip(self.info_keys, info_values):
            episode['info_{}'.format(key)] = value

        # stats
        successful = np.array(successes)[-1, :]
        assert successful.shape == (self.rollout_batch_size,)
        success_rate = np.mean(successful)
        self.success_history.append(success_rate)

        successful_any = np.array(successes).max(0)
        assert successful_any.shape == (self.rollout_batch_size,)
        success_rate = np.mean(successful_any)
        self.success_any_history.append(success_rate)

        if self.compute_Q:
            self.Q_history.append(np.mean(Qs))
        self.n_episodes += self.rollout_batch_size

        return convert_episode_to_batch_major(episode)

    def clear_history(self):
        """Clears all histories that are used for statistics
        """
        self.success_history.clear()
        self.success_any_history.clear()
        self.Q_history.clear()

    def current_success_rate(self):
        return np.mean(self.success_history)

    def current_mean_Q(self):
        return np.mean(self.Q_history)

    def save_policy(self, path):
        """Pickles the current policy for later inspection.
        """
        with open(path, 'wb') as f:
            pickle.dump(self.policy, f)

    def logs(self, prefix='worker'):
        """Generates a dictionary that contains all collected statistics.
        """
        logs = []
        logs += [('success_rate', np.mean(self.success_history))]
        logs += [('success_any_rate', np.mean(self.success_any_history))]
        if self.compute_Q:
            logs += [('mean_Q', np.mean(self.Q_history))]
        logs += [('episode', self.n_episodes)]

        if prefix is not '' and not prefix.endswith('/'):
            return [(prefix + '/' + key, val) for key, val in logs]
        else:
            return logs

    def seed(self, seed):
        """Seeds each environment with a distinct seed derived from the passed in global seed.
        """
        for idx, env in enumerate(self.envs):
            env.seed(seed + 1000 * idx)
