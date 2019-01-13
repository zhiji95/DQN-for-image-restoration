import random
import numpy as np


class SumTree(object):
    """
    This SumTree code is a modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py
    Story data with its priority in the tree.
    """
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:  # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:  # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1  # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):  # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:  # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], data_idx, self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root


class PReplayMemory:
    def __init__(self, config):
        # sumtree
        self.tree = SumTree(4000)
        self.epsilon = 0.01  # small amount to avoid zero priority
        self.alpha = 0.6  # [0~1] convert the importance of TD error to priority
        self.beta = 0.4  # importance-sampling, from initial value increasing to 1
        self.beta_increment_per_sampling = 0.001
        self.abs_err_upper = 1.  # clipped abs error

        self.config = config
        self.memory_size = config.memory_size
        self.actions = np.empty(self.memory_size, dtype=np.uint8)
        self.rewards = np.empty(self.memory_size, dtype=np.float16)
        self.screens = np.empty((self.memory_size, config.screen_height, config.screen_width, config.screen_channel),
                                dtype=np.float16)
        self.terminals = np.empty(self.memory_size, dtype=np.bool)
        self.history_length = 1
        self.dims = (config.screen_height, config.screen_width, config.screen_channel)
        self.batch_size = config.batch_size
        self.count = 0
        self.current = 0
        self.stop_step = config.stop_step
        self.safe_length = self.stop_step + 1

        # pre-allocate prestates and poststates for minibatch
        self.prestates = np.empty((self.batch_size, self.history_length) + self.dims, dtype=np.float16)
        self.poststates = np.empty((self.batch_size, self.history_length) + self.dims, dtype=np.float16)

    def add(self, screen, reward, action, terminal):
        screen_temp = screen.reshape(screen.shape[1:])
        assert screen_temp.shape == self.dims
        # NB! screen is post-state, after action and reward
        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.screens[self.current, ...] = screen_temp
        self.terminals[self.current] = terminal
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.memory_size
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        #         print(screen_temp, action,reward,terminal)
        #         transition = np.hstack((screen_temp, [action, reward],terminal))
        self.tree.add(max_p, (screen, reward, action, terminal))

    #     def sample(self,n):
    #         b_idx, ISWeights = np.empty((n,),dtype=np.int32), np.empty((n,1))
    #         actions = np.empty(n, dtype = np.uint8)
    #         rewards = np.empty(n, dtype = np.float16)
    #         screens = np.empty((n, self.config.screen_height,self.config.screen_width, self.config.screen_channel), dtype = np.float16)
    #         terminals = np.empty(n, dtype = np.bool)

    #         pri_seg = self.tree.total_p / n

    #         self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

    #         min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p  # for later calculate ISweight

    #         for i in range(n):
    #             a, b = pri_seg * i, pri_seg * (i + 1)
    #             v = np.random.uniform(a, b)
    #             idx, p,data_idx, data = self.tree.get_leaf(v)
    #             prob = p / self.tree.total_p
    #             ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
    #             b_idx[i] = idx
    #             j = data_idx + 1
    #             while not self.terminals[i]:
    #                 j += 1
    #             screens[i] = data[0]
    #             actions[i] = data[1]
    #             rewards[i] = data[2]
    #             terminals[i] = data[3]
    #         return screens,actions,rewards, b_idx, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

    def getEpiBatch(self, batch_size):
        s_t = []
        action = []
        reward = []
        n = batch_size
        b_idx, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, 1))
        pri_seg = self.tree.total_p / n
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])
        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p  # for later calculate ISweight

        # episode may have different lengths
        for _ in range(self.stop_step):
            s_t.append([])
            action.append([])
            reward.append([])
        for i in range(batch_size):
            a, b = pri_seg * i, pri_seg * (i + 1)
            itera = 0
            while True:
                #             index = random.randint(self.history_length + self.safe_length, self.count - 1)
                #             if wraps over current pointer, then get new one
                if itera > 50000:
                    print("large iteration, random select")
                    index = random.randint(self.history_length + self.safe_length, self.count - 1)
                else:
                    v = np.random.uniform(a, b)
                    idx, p, index, data = self.tree.get_leaf(v)
                itera += 1
                if index - self.history_length - self.safe_length <= self.current <= \
                        index + self.history_length + self.safe_length:
                    continue
                # if wraps over episode end, then get new one
                elif self.terminals[(index - self.history_length):index].any():
                    continue
                # in case touch the end
                elif index + self.history_length + self.safe_length >= self.memory_size or \
                        index - self.history_length - self.safe_length <= 0:
                    continue

                else:
                    break

            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob / min_prob, -self.beta)
            b_idx[i] = idx

            cur_episode = self.getEpisode(index)
            len_cur = len(cur_episode)  # length of current episode
            s_t_cur = s_t[len_cur - 1]
            action_cur = action[len_cur - 1]
            reward_cur = reward[len_cur - 1]
            for m in range(len_cur):
                s_t_cur.append(cur_episode[m, 0])
                action_cur.append(cur_episode[m, 1])
                reward_cur.append(cur_episode[m, 2])
        for k in range(self.stop_step):
            if len(reward[k]) > 0:
                s_t[k] = np.concatenate(s_t[k], axis=0).astype(np.float)
            action[k] = np.array(action[k], dtype=np.int)
            reward[k] = np.array(reward[k], dtype=np.float)

        return s_t, action, reward, b_idx, ISWeights

    #     def getEpiBatch(self, batch_size):
    #         s_t = []
    #         action = []
    #         reward = []
    #         # episode may have different lengths
    #         for _ in range(self.stop_step):
    #             s_t.append([])
    #             action.append([])
    #             reward.append([])
    #         for _ in range(batch_size):
    #             cur_episode = self.getEpisode()
    #             len_cur = len(cur_episode)  # length of current episode
    #             s_t_cur = s_t[len_cur - 1]
    #             action_cur = action[len_cur - 1]
    #             reward_cur = reward[len_cur - 1]
    #             for m in range(len_cur):
    #                 s_t_cur.append(cur_episode[m, 0])
    #                 action_cur.append(cur_episode[m, 1])
    #                 reward_cur.append(cur_episode[m, 2])
    #         for k in range(self.stop_step):
    #             if len(reward[k]) > 0:
    #                 s_t[k] = np.concatenate(s_t[k], axis=0).astype(np.float)
    #             action[k] = np.array(action[k], dtype=np.int)
    #             reward[k] = np.array(reward[k], dtype=np.float)

    #         return s_t, action, reward

    def getEpisode(self, index):  # return single episode
        assert self.count > self.history_length
        # search for the start state
        idx_start = index
        while not self.terminals[idx_start - 2]:
            idx_start -= 1
        # search for the end state
        idx_end = index
        while not self.terminals[idx_end]:
            idx_end += 1
        #         print(index, idx_end, self.stop_step,idx_start)
        #         idx_end = min(idx_end, self.stop_step+idx_start-1)
        # get the whole episode
        output = []
        for k in range(idx_start, idx_end + 1):
            s_t = self.getState(k - 1).copy()
            action = self.actions[k]
            reward = self.rewards[k]
            s_t_plus_1 = self.getState(k).copy()
            terminals = self.terminals[k]
            output.append([s_t, action, reward, s_t_plus_1, terminals])
        output = np.array(output)

        assert output[-1, -1]
        return output

    def getState(self, index):
        assert self.count > 0
        # normalize index to expected range, allows negative indexes
        index = index % self.count
        # if is not in the beginning of matrix
        if index >= self.history_length - 1:
            # use faster slicing
            return self.screens[(index - (self.history_length - 1)):(index + 1), ...]
        else:
            # otherwise normalize indexes and use slower list based access
            indexes = [(index - i) % self.count for i in reversed(range(self.history_length))]
            return self.screens[indexes, ...]


class ReplayMemory:
    def __init__(self, config):
        self.memory_size = config.memory_size
        self.actions = np.empty(self.memory_size, dtype=np.uint8)
        self.rewards = np.empty(self.memory_size, dtype=np.float16)
        self.screens = np.empty((self.memory_size, config.screen_height, config.screen_width, config.screen_channel),
                                dtype=np.float16)
        self.terminals = np.empty(self.memory_size, dtype=np.bool)
        self.history_length = 1
        self.dims = (config.screen_height, config.screen_width, config.screen_channel)
        self.batch_size = config.batch_size
        self.count = 0
        self.current = 0
        self.stop_step = config.stop_step
        self.safe_length = self.stop_step + 1

        # pre-allocate prestates and poststates for minibatch
        self.prestates = np.empty((self.batch_size, self.history_length) + self.dims, dtype=np.float16)
        self.poststates = np.empty((self.batch_size, self.history_length) + self.dims, dtype=np.float16)

    def add(self, screen, reward, action, terminal):
        screen_temp = screen.reshape(screen.shape[1:])
        assert screen_temp.shape == self.dims
        # NB! screen is post-state, after action and reward
        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.screens[self.current, ...] = screen_temp
        self.terminals[self.current] = terminal
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.memory_size

    def getEpiBatch(self, batch_size):
        s_t = []
        action = []
        reward = []
        # episode may have different lengths
        for _ in range(self.stop_step):
            s_t.append([])
            action.append([])
            reward.append([])
        for _ in range(batch_size):
            cur_episode = self.getEpisode()
            len_cur = len(cur_episode)  # length of current episode
            s_t_cur = s_t[len_cur - 1]
            action_cur = action[len_cur - 1]
            reward_cur = reward[len_cur - 1]
            for m in range(len_cur):
                s_t_cur.append(cur_episode[m, 0])
                action_cur.append(cur_episode[m, 1])
                reward_cur.append(cur_episode[m, 2])
        for k in range(self.stop_step):
            if len(reward[k]) > 0:
                s_t[k] = np.concatenate(s_t[k], axis=0).astype(np.float)
            action[k] = np.array(action[k], dtype=np.int)
            reward[k] = np.array(reward[k], dtype=np.float)

        return s_t, action, reward

    def getEpisode(self):  # return single episode
        assert self.count > self.history_length
        while True:
            index = random.randint(self.history_length + self.safe_length, self.count - 1)
            # if wraps over current pointer, then get new one
            if index - self.history_length - self.safe_length <= self.current <= \
                    index + self.history_length + self.safe_length:
                continue
            # if wraps over episode end, then get new one
            if self.terminals[(index - self.history_length):index].any():
                continue
            # in case touch the end
            if index + self.history_length + self.safe_length >= self.memory_size or \
                    index - self.history_length - self.safe_length <= 0:
                continue

            # search for the start state
            idx_start = index
            while not self.terminals[idx_start - 2]:
                idx_start -= 1
            # search for the end state
            idx_end = index
            while not self.terminals[idx_end]:
                idx_end += 1

            # get the whole episode
            output = []
            for k in range(idx_start, idx_end + 1):
                s_t = self.getState(k - 1).copy()
                action = self.actions[k]
                reward = self.rewards[k]
                s_t_plus_1 = self.getState(k).copy()
                terminals = self.terminals[k]
                output.append([s_t, action, reward, s_t_plus_1, terminals])
            output = np.array(output)
            assert output[-1, -1]
            return output

    def getState(self, index):
        assert self.count > 0
        # normalize index to expected range, allows negative indexes
        index = index % self.count
        # if is not in the beginning of matrix
        if index >= self.history_length - 1:
            # use faster slicing
            return self.screens[(index - (self.history_length - 1)):(index + 1), ...]
        else:
            # otherwise normalize indexes and use slower list based access
            indexes = [(index - i) % self.count for i in reversed(range(self.history_length))]
            return self.screens[indexes, ...]





