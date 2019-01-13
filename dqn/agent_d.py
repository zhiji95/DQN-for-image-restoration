import cv2
import numpy as np
import os
import random
import scipy as sci
import scipy.io as sio
import tensorflow as tf
from .base import BaseModel
from .ops import linear, conv2d, clipped_error
from .replay_memory import ReplayMemory
from .replay_memory import PReplayMemory
from .utils import data_reformat, img2patch
from tqdm import tqdm
import pandas as pd


class Agent(BaseModel):
    def __init__(self, config, environment, sess):
        super(Agent, self).__init__(config)
        self.sess = sess
        self.env = environment
        self.cnn_format = 'NHWC'
        self.action_size = self.env.action_size
        self.build_dqn()
        self.stat = {}
        if self.is_train:
            self.memory = PReplayMemory(self.config)

    def build_dqn(self):
        self.w = {}
        self.t_w = {}

        # training network
        activation_fn = tf.nn.relu
        with tf.variable_scope('prediction'):
            # input batch and length for recurrent training
            self.batch = tf.placeholder(tf.int32, shape=[])
            self.length = tf.placeholder(tf.int32)
            self.s_t = tf.placeholder('float32', [None, self.screen_height, self.screen_width, self.screen_channel],
                                      name='s_t')
            self.l1, self.w['l1_w'], self.w['l1_b'] = conv2d(self.s_t, 32, [9, 9], [2, 2],
                                                             activation_fn, self.cnn_format, name='l1')
            self.l1_out = self.l1
            self.l2, self.w['l2_w'], self.w['l2_b'] = conv2d(self.l1_out, 24, [5, 5], [2, 2],
                                                             activation_fn, self.cnn_format, name='l2')
            self.l2_out = self.l2
            self.l3, self.w['l3_w'], self.w['l3_b'] = conv2d(self.l2_out, 24, [5, 5], [2, 2],
                                                             activation_fn, self.cnn_format, name='l3')
            self.l3_out = self.l3
            self.l4, self.w['l4_w'], self.w['l4_b'] = conv2d(self.l3_out, 24, [5, 5], [2, 2],
                                                             activation_fn, self.cnn_format, name='l4')
            shape = self.l4.get_shape().as_list()
            len_flat = 1
            for k in range(1, len(shape)):
                len_flat *= shape[k]
            self.l6_flat = tf.reshape(self.l4, [-1, len_flat])

            # Add action as input
            self.action_in = tf.placeholder('float32', [None, self.action_size - 1], name='action_in')
            self.action_out = self.action_in
            self.l7, self.w['l7_w'], self.w['l7_b'] = linear(self.l6_flat, self.lstm_in,
                                                             activation_fn=activation_fn, name='l7')
            self.l7_action = tf.concat([self.l7, self.action_out], 1)

            # add LSTM with dynamic steps
            self.rnn_input = tf.reshape(self.l7_action, [self.batch, self.length,
                                                         self.l7_action.get_shape().as_list()[-1]])
            self.rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.h_size, state_is_tuple=True)
            self.state_in = self.rnn_cell.zero_state(self.batch, tf.float32)
            self.rnn, self.rnn_state = tf.nn.dynamic_rnn(inputs=self.rnn_input, cell=self.rnn_cell, dtype=tf.float32,
                                                         initial_state=self.state_in, scope='prediction_rnn')
            self.rnn = tf.reshape(self.rnn, shape=[-1, self.h_size])
            self.w['rnn_w'], self.w['rnn_b'] = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                                 scope='prediction/prediction_rnn')
            # end LSTM

            # Construct Dueling network
            #             self.V,_,_ = linear(self.rnn, 1, name='V')
            #             self.A, self.w['q_w'], self.w['q_b'] = linear(self.rnn, self.action_size, name='A')
            #             self.q = self.V + self.A - tf.reduce_mean(self.A,axis=1,keep_dims=True)
            self.q, self.w['q_w'], self.w['q_b'] = linear(self.rnn, self.action_size, name='q')
            self.q_action = tf.argmax(self.q, axis=1)

        if self.is_train:
            # target network
            with tf.variable_scope('target'):
                # input batch and length for recurrent training
                self.t_batch = tf.placeholder(tf.int32, shape=[])
                self.t_length = tf.placeholder(tf.int32, shape=[])

                self.target_s_t = tf.placeholder('float32', [None, self.screen_height,
                                                             self.screen_width, self.screen_channel], name='target_s_t')
                self.target_l1, self.t_w['l1_w'], self.t_w['l1_b'] = \
                    conv2d(self.target_s_t, 32, [9, 9], [2, 2], activation_fn, self.cnn_format, name='target_l1')
                self.target_l1_out = self.target_l1
                self.target_l2, self.t_w['l2_w'], self.t_w['l2_b'] = \
                    conv2d(self.target_l1_out, 24, [5, 5], [2, 2], activation_fn, self.cnn_format, name='target_l2')
                self.target_l2_out = self.target_l2
                self.target_l3, self.t_w['l3_w'], self.t_w['l3_b'] = \
                    conv2d(self.target_l2_out, 24, [5, 5], [2, 2], activation_fn, self.cnn_format, name='target_l3')
                self.target_l3_out = self.target_l3
                self.target_l4, self.t_w['l4_w'], self.t_w['l4_b'] = \
                    conv2d(self.target_l3_out, 24, [5, 5], [2, 2], activation_fn, self.cnn_format, name='target_l4')
                shape = self.target_l4.get_shape().as_list()
                target_len_flat = 1
                for k in range(1, len(shape)):
                    target_len_flat *= shape[k]
                self.target_l6_flat = tf.reshape(self.target_l4, [-1, target_len_flat])

                # Add action as input
                self.target_action_in = tf.placeholder('float32', [None, self.action_size - 1],
                                                       name='target_action_in')
                self.target_action_out = self.target_action_in
                self.target_l7, self.t_w['l7_w'], self.t_w['l7_b'] = \
                    linear(self.target_l6_flat, self.lstm_in, activation_fn=activation_fn, name='target_l7')
                self.target_l7_action = tf.concat([self.target_l7, self.target_action_out], 1)

                # add LSTM with dynamic steps
                self.t_rnn_input = tf.reshape(self.target_l7_action, [self.t_batch, self.t_length,
                                                                      self.target_l7_action.get_shape().as_list()[-1]])
                self.t_rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.h_size, state_is_tuple=True)
                self.t_state_in = self.t_rnn_cell.zero_state(self.t_batch, tf.float32)
                self.t_rnn, self.t_rnn_state = tf.nn.dynamic_rnn(inputs=self.t_rnn_input, cell=self.t_rnn_cell,
                                                                 dtype=tf.float32, initial_state=self.t_state_in,
                                                                 scope='target_rnn')
                self.t_rnn = tf.reshape(self.t_rnn, shape=[-1, self.h_size])
                self.t_w['rnn_w'], self.t_w['rnn_b'] = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                                         scope='target/target_rnn')
                # end LSTM
                # Construct Dueling network
                #                 self.t_V,_,_ = linear(self.t_rnn, 1, name='target_V')
                #                 self.t_A, self.t_w['q_w'], self.t_w['q_b'] = linear(self.t_rnn, self.action_size, name='target_A')
                #                 self.target_q = self.t_V + self.t_A - tf.reduce_mean(self.t_A,axis=1,keep_dims=True)

                self.target_q, self.t_w['q_w'], self.t_w['q_b'] = \
                    linear(self.t_rnn, self.action_size, name='target_q')
                self.target_q_idx = tf.placeholder('int32', [None, None], 'outputs_idx')
                self.target_q_with_idx = tf.gather_nd(self.target_q, self.target_q_idx)

                # update target network
            with tf.variable_scope('pred_to_target'):
                self.t_w_input = {}
                self.t_w_assign_op = {}

                for name in self.w.keys():
                    self.t_w_input[name] = tf.placeholder('float32', self.t_w[name].get_shape().as_list(), name=name)
                    self.t_w_assign_op[name] = self.t_w[name].assign(self.t_w_input[name])

            # optimizer
            with tf.variable_scope('optimizer'):
                self.global_step = tf.Variable(0, trainable=False)
                self.target_q_t = tf.placeholder('float32', [None], name='target_q_t')
                self.ISWeights = tf.placeholder(tf.float32, [None, 1], name='IS_weights')

                self.action = tf.placeholder('int64', [None], name='action')
                action_one_hot = tf.one_hot(self.action, self.action_size, 1.0, 0.0, name='action_one_hot')
                q_acted = tf.reduce_sum(self.q * action_one_hot, reduction_indices=1, name='q_acted')
                self.delta = self.target_q_t - q_acted
                #                 self.loss = tf.reduce_mean(clipped_error(self.delta), name='loss')

                #                 self.loss = tf.reduce_mean(self.ISWeights * clipped_error(self.delta), name='loss')

                self.abs_errors = tf.reduce_sum(tf.abs(self.target_q - self.q), axis=1)  # for updating Sumtree
                #                 self.loss = tf.reduce_mean(self.ISWeights * tf.squared_difference(self.target_q_t, q_acted), name='loss')

                #                 self.loss = tf.reduce_mean(1/(self.ISWeights) * clipped_error(self.delta), name='loss')

                self.loss = tf.reduce_sum(self.ISWeights * clipped_error(self.delta), name='loss')

                self.learning_rate_step = tf.placeholder('int64', None, name='learning_rate_step')
                self.learning_rate_op = tf.maximum(self.learning_rate_minimum,
                                                   tf.train.exponential_decay(
                                                       self.learning_rate,
                                                       self.learning_rate_step,
                                                       self.learning_rate_decay_step,
                                                       self.learning_rate_decay,
                                                       staircase=False))
                self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate_op)

                # accumulated gradients (for training LSTM with toolchains of dynamic length)
                tvs = list(self.w.values())
                accum_vars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvs]
                self.zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars]
                gvs = self.opt.compute_gradients(self.loss, tvs)
                self.accum_weight = tf.placeholder(tf.float32, shape=[])  # gradient weight
                self.accum_ops = [accum_vars[i].assign_add(gv[0] * self.accum_weight) for i, gv in enumerate(gvs)]
                self.train_step = self.opt.apply_gradients([(accum_vars[i], gv[1]) for i, gv in enumerate(gvs)])

            with tf.variable_scope('summary'):
                scalar_summary_tags = ['training.learning_rate']
                reward_tags = ['test.reward' + str(x + 1) for x in range(self.stop_step)] + ['test.reward_sum']
                scalar_summary_tags += reward_tags + reward_tags

                self.summary_placeholders = {}
                self.summary_ops = {}

                for tag in scalar_summary_tags:
                    self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_'))
                    self.summary_ops[tag] = tf.summary.scalar("%s" % tag, self.summary_placeholders[tag])

                self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

        tf.global_variables_initializer().run()
        if not self.is_train:
            self.load_model()

    def train(self):
        start_step, self.update_count = 0, 0
        ep_reward, total_reward, self.total_loss, self.total_q = 0., 0., 0., 0.
        max_avg_ep_reward = -1
        total_rewards, avg_losses, avg_qs = [], [], []
        rsum, rsum1, rsum2, rsum3 = [], [], [], []
        ep_rewards, actions = [], []
        img, reward, action, terminal = self.env.new_image()  # get the first input image
        self.state_explore = (np.zeros([1, self.h_size]), np.zeros([1, self.h_size]))  # initialize LSTM states
        #         for self.step in tqdm(range(start_step, 50000), ncols=70, initial=start_step):

        for self.step in tqdm(range(start_step, self.max_step), ncols=70, initial=start_step):
            if self.step % 50000 == 0:
                self.stat["reward sum"] = rsum
                self.stat["average loss"] = avg_losses
                self.stat["average Q"] = avg_qs
                df = pd.DataFrame(self.stat)
                df.to_csv("./checkpoint" + str(self.step) + ".csv")

            if self.step == self.learn_start:
                self.update_count, ep_reward = 0, 0.
                total_reward, self.total_loss, self.total_q = 0., 0., 0.
                ep_rewards, actions = [], []

            # 1. agent predicts an action
            action, ep = self.predict(img)

            self.pre_action = action

            # 2. environment conducts the action
            img, reward, terminal = self.env.act(action)

            # 3. observe (add memory, train)
            self.observe(img, reward, action, terminal)

            if terminal:
                img, _, _, terminal = self.env.new_image()
                self.state_explore = (np.zeros([1, self.h_size]), np.zeros([1, self.h_size]))  # initialize LSTM states
                ep_reward += reward
                ep_rewards.append(ep_reward)
                ep_reward = 0.

                # add the initial state
                self.memory.add(img, 0., -1, terminal)  # action -1=255 for uint8

            else:
                ep_reward += reward

            actions.append(action)
            # Calculate and record data
            total_reward += reward

            # validation and logging
            if self.step >= self.learn_start and self.step % self.test_step == self.test_step - 1:
                avg_reward = total_reward / self.test_step
                avg_loss = self.total_loss / self.update_count
                avg_q = self.total_q / self.update_count
                avg_ep_reward = np.mean(ep_rewards)

                #                 print('\navg_r: %.4f, avg_l: %.6f, avg_q: %3.6f, avg_ep_r: %.4f' \
                #                       % (avg_reward, avg_loss, avg_q, avg_ep_reward))

                if self.step % self.save_step == self.save_step - 1 and max_avg_ep_reward * 0.9 <= avg_ep_reward:
                    self.save_model(self.step + 1)
                    max_avg_ep_reward = max(max_avg_ep_reward, avg_ep_reward)

                # validation
                reward_test_vec = []
                psnr_test_vec = []
                reward_str = ''
                psnr_str = ''
                reward_sum = 0.
                for cur_step in range(self.stop_step):
                    action_test = self.predict_test(count_step=cur_step)
                    self.pre_action_test = action_test.copy()
                    reward_test, psnr_test, base_psnr = self.env.act_test(action_test, step=cur_step)
                    reward_sum += reward_test
                    reward_test_vec.append(reward_test)
                    psnr_test_vec.append(psnr_test)
                    reward_str += 'reward' + str(cur_step + 1) + ': %.4f, '
                    psnr_str += 'psnr' + str(cur_step + 1) + ': %.4f, '
                reward_str += 'reward_sum: %.4f, '
                reward_test_vec.append(reward_sum)
                print_str = reward_str + psnr_str + 'base_psnr: %.4f'
                print(print_str % tuple(reward_test_vec + psnr_test_vec + [base_psnr]))

                rsum.append(reward_sum)
                #                 rsums.append(reward_test_vec)
                avg_losses.append(avg_loss)
                avg_qs.append(avg_q)

                # logging (write summary)
                diction = {'training.learning_rate':
                               self.learning_rate_op.eval({self.learning_rate_step: self.step})}
                for cur_step in range(self.stop_step):
                    diction['test.reward' + str(cur_step + 1)] = reward_test_vec[cur_step]
                diction['test.reward_sum'] = reward_test_vec[-1]
                self.inject_summary(diction)

                total_reward = 0.
                self.total_loss = 0.
                self.total_q = 0.
                self.update_count = 0
                ep_reward = 0.
                ep_rewards = []
                actions = []
        self.stat["reward sum"] = rsum
        self.stat["average loss"] = avg_losses
        self.stat["average Q"] = avg_qs

    def predict(self, s_t, test_ep=None):
        action_size = self.action_size
        ep = test_ep or (self.ep_end +
                         max(0., (self.ep_start - self.ep_end)
                             * (self.ep_end_t - max(0., self.step - self.learn_start)) / self.ep_end_t))
        if random.random() < ep:
            action = random.randrange(action_size)
        else:
            count_step = self.env.count
            action_in = np.zeros([1, action_size - 1])
            if count_step > 0:
                action_in[0, self.pre_action] = 1.
            action_vec, self.state_explore = self.sess.run([self.q, self.rnn_state],
                                                           {self.s_t: s_t, self.action_in: action_in,
                                                            self.batch: 1, self.length: 1,
                                                            self.state_in: self.state_explore})
            action_vec_target = self.sess.run(self.target_q,
                                              {self.target_s_t: s_t, self.target_action_in: action_in,
                                               self.t_batch: 1, self.t_length: 1})
            action_vec = action_vec[0]
            action_vec_target = action_vec_target[0]
            action = np.argmax([sum(x) for x in zip(action_vec, action_vec_target)], axis=0)

        #             action = action_vec.argmax(axis=0)

        return action, ep

    def observe(self, screen, reward, action, terminal):
        # add memory
        self.memory.add(screen, reward, action, terminal)

        if self.step > self.learn_start:
            # train agent network
            if self.step % self.train_frequency == 0:
                self.q_learning_lstm_batch()

            # update target network
            if self.step % self.target_q_update_step == self.target_q_update_step - 1:
                self.update_target_q_network()

    def q_learning_lstm_batch(self):
        # assign accumulated gradients to zero
        sess = tf.get_default_session()
        sess.run(self.zero_ops)

        # initialization
        action_size = self.action_size
        temp_q_t = 0.
        temp_loss = 0.

        # get a batch of episodes (toolchains)
        #         s_t, action, reward = self.memory.getEpiBatch(self.batch_size)
        # get a batch of sumtree information
        s_t, action, reward, tree_idx, ISWeights = self.memory.getEpiBatch(self.batch_size)
        #         print(ISWeights)
        # loop for different lengths of episode
        for m in range(self.stop_step):
            s_t_cur = s_t[m]
            action_cur = action[m]
            reward_cur = reward[m]
            num = len(reward_cur)
            if num < 1:
                continue
            rnn_length = m + 1
            rnn_batch = num // rnn_length

            # derive action at previous step
            action_in = np.zeros([num, action_size - 1])
            for k in range(rnn_batch):
                for p in range(rnn_length):
                    if p > 0:
                        idx = k * rnn_length + p
                        action_in[idx, action_cur[idx - 1]] = 1.

            # target Q value
            q_t_plus_1 = self.q.eval({self.s_t: s_t_cur, self.action_in: action_in,
                                      self.batch: rnn_batch, self.length: rnn_length})
            max_q_t_plus_1 = np.argmax(q_t_plus_1, axis=1)

            target_temp = self.target_q.eval({self.target_s_t: s_t_cur, self.target_action_in: action_in,
                                              self.t_batch: rnn_batch, self.t_length: rnn_length})
            target_new = []
            for i, l in enumerate(target_temp):
                target_new.append(l[max_q_t_plus_1[i]])
            for k in range(rnn_batch):
                idx = k * rnn_length
                target_new[idx: idx + rnn_length - 1] = target_new[idx + 1: idx + rnn_length]
                target_new[idx + rnn_length - 1] = 0.
            # accumulate gradients
            target_q_t = self.discount * np.array(target_new) + reward_cur
            grad_weight = 1.0 * rnn_batch / self.batch_size  # gradient weight
            q_t, abs_errors, loss, _ = sess.run([self.q, self.abs_errors, self.loss, self.accum_ops],
                                                {self.target_q_t: target_q_t,
                                                 self.action: action_cur,
                                                 self.s_t: s_t_cur,
                                                 self.learning_rate_step: self.step,
                                                 self.action_in: action_in,
                                                 self.batch: rnn_batch,
                                                 self.length: rnn_length,
                                                 self.accum_weight: grad_weight,
                                                 self.ISWeights: ISWeights,
                                                 self.target_s_t: s_t_cur,
                                                 self.target_action_in: action_in,
                                                 self.t_batch: rnn_batch,
                                                 self.t_length: rnn_length
                                                 })
            temp_q_t += q_t.mean()
            temp_loss += loss

        #             # accumulated cost
        #             _, abs_errors, loss = self.sess.run([self._train_op, self.abs_errors, self.loss],
        #                                          feed_dict={self.s: batch_memory[:, :self.n_features],
        #                                                     self.q_target: q_target,
        #                                                     self.ISWeights: ISWeights})
        self.memory.batch_update(tree_idx, abs_errors)  # update priority

        # apply gradients
        sess.run(self.train_step, {self.learning_rate_step: self.step})

        # update statistics
        self.total_loss += temp_loss
        self.total_q += q_t.mean() / self.batch_size
        self.update_count += 1

    def update_target_q_network(self):
        for name in self.w.keys():
            self.t_w_assign_op[name].eval({self.t_w_input[name]: self.w[name].eval()})

    #         for name in self.w.keys():
    #             self.eval_w_assign_op[name].eval({self.eval_w_input[name]: self.weval[name].eval()})

    def inject_summary(self, tag_dict):
        summary_str_lists = self.sess.run([self.summary_ops[tag] for tag in tag_dict.keys()],
                                          {self.summary_placeholders[tag]: value for tag, value in tag_dict.items()})
        for summary_str in summary_str_lists:
            self.writer.add_summary(summary_str, self.step + 1)

    def predict_test(self, count_step=0):
        if count_step == 0:
            imgs = self.env.get_data_test()
            env_steps = np.zeros(len(imgs), dtype=int)
            # initialize LSTM states
            self.state_test = (np.zeros([len(imgs), self.h_size]), np.zeros([len(imgs), self.h_size]))
            self.sess_test = tf.get_default_session()
        else:
            imgs = self.env.get_test_imgs()
            env_steps = self.env.get_test_steps()

        action_in = np.zeros([len(imgs), self.action_size - 1])
        if count_step > 0:
            for k in range(len(imgs)):
                if self.pre_action_test[k] < self.action_size - 1:
                    action_in[k, self.pre_action_test[k]] = 1.

        actions_vec, self.state_test = self.sess_test.run([self.q, self.rnn_state],
                                                          {self.s_t: imgs, self.action_in: action_in,
                                                           self.state_in: self.state_test,
                                                           self.batch: len(imgs), self.length: 1})
        actions = actions_vec.argmax(axis=1)

        # choose the last action if pre-action is the last
        if count_step > 0:
            actions[self.pre_action_test == self.action_size - 1] = self.action_size - 1
        actions[env_steps == self.stop_step] = self.action_size - 1  # already stopped before
        # choose the last action if pre-action is the last
        return actions




