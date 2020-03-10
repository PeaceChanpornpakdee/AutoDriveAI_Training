import tensorflow as tf
import tensorflow.compat.v1 as tfc
import numpy as np
#import gym
import time
import sys
from V_06 import VREP_env

tfc.disable_eager_execution()
class Network(object):


    def __init__(self, env, scope, num_layers, num_units, obs_plc, act_plc, trainable=True):
        self.env = env
        #self.observation_size = 24
        self.action_size = 2
        self.trainable = trainable
        self.scope = scope

        self.obs_place = obs_plc
        self.acts_place = act_plc

        self.p, self.v, self.logstd = self._build_network(num_layers=num_layers, num_units=num_units)
        self.act_op = self.action_sample()

    def _build_network(self, num_layers, num_units):  #More from P'Sumo
        with tfc.variable_scope(self.scope):
            x = self.obs_place

            logstd = tfc.get_variable(name="logstd", shape=[self.action_size],
                                     initializer=tf.zeros_initializer)

            for i in range(num_layers):
                x = tfc.layers.dense(x, units=num_units, activation=tf.nn.relu, name="px_fc" + str(i),
                                    trainable=self.trainable, use_bias=False)
            action = tfc.layers.dense(x, units=self.action_size, activation=tf.tanh,
                                     name="p_fc" + str(num_layers), trainable=self.trainable, use_bias=False)


            x = self.obs_place
            for i in range(num_layers):
                x = tfc.layers.dense(x, units=num_units, activation=tf.nn.relu, name="v_fc" + str(i),
                                    trainable=self.trainable, use_bias=False)
            value = tfc.layers.dense(x, units=1, activation=None, name="v_fc" + str(num_layers),
                                    trainable=self.trainable, use_bias=False)


        return action, value, logstd

    def action_sample(self):
        return self.p + 1.0*(tf.exp(self.logstd) * tfc.random_normal(tf.shape(self.p)))

    def get_variables(self):
        return tfc.get_collection(tfc.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tfc.get_collection(tfc.GraphKeys.TRAINABLE_VARIABLES, self.scope)


class PPOAgent(object):
    def __init__(self, env):
        self.env = env

        # hyperparameters
        self.learning_rate = 1e-4 
        self.epochs = 10
        self.n_round = 15
        #self.step_size = 3072
        self.step_size = 500*self.n_round #<**>
        self.gamma = 0.99 #<**>
        self.lam = 0.95
        ###self.clip_param = 0.05
        self.clip_param = 0.2 #<**>
        ###self.batch_size = 64
        self.batch_size = 500 #<**>

        # placeholders
        self.adv_place = tfc.placeholder(shape=[None], dtype=tf.float32)
        self.return_place = tfc.placeholder(shape=[None], dtype=tf.float32)

        ### self.obs_place = tfc.placeholder(shape=[None, env.observation_space.shape[0]],
        ###                                 name='ob', dtype=tf.float32)
        self.obs_place = tfc.placeholder(shape=[None, 21],
                                        name='ob', dtype=tf.float32)
        ### self.acts_place = tfc.placeholder(shape=[None, env.action_space.shape[0]],
        ###                                  name='ac', dtype=tf.float32)
        self.acts_place = tfc.placeholder(shape=[None, 2],
                                         name='ac', dtype=tf.float32)

        # build network
        self.net = Network(env=self.env,
                           scope="pi",
                           num_layers=4,
                           num_units=64,
                           obs_plc=self.obs_place,
                           act_plc=self.acts_place)

        self.old_net = Network(env=self.env,
                               scope="old_pi",
                               num_layers=4,
                               num_units=64,
                               obs_plc=self.obs_place,
                               act_plc=self.acts_place,
                               trainable=False)

        # tensorflow operators
        self.assign_op = self.assign(self.net, self.old_net)
        self.ent, self.pol_loss, self.vf_loss, self.update_op = self.update()
        self.saver = tfc.train.Saver()

    @staticmethod
    def logp(net):
        logp = -(0.5 * tf.reduce_sum(tf.square((net.acts_place - net.p) / tf.exp(net.logstd)), axis=-1) \
                 + 0.5 * np.log(2.0 * np.pi) * tfc.to_float(tf.shape(net.p)[-1]) \
                 + tf.reduce_sum(net.logstd, axis=-1))

        return logp

    @staticmethod
    def entropy(net):
        ent = tf.reduce_sum(net.logstd + .5 * np.log(2.0 * np.pi * np.e), axis=-1)
        return ent

    @staticmethod
    def assign(net, old_net):
        assign_op = []
        for (newv, oldv) in zip(net.get_variables(), old_net.get_variables()):
            assign_op.append(tfc.assign(oldv, newv))

        return assign_op

    def traj_generator(self):
        t = 0
        ###action = env.action_space.sample()
        ###done = True
        ###ob = env.reset()
        env.setUp()
        
        action = [0,0]
        done = False
        ob = env.observe()


        cur_ep_return = 0
        cur_ep_length = 0
        ep_returns = []
        ep_lengths = []

        obs = np.array([ob for _ in range(self.step_size)])
        rewards = np.zeros(self.step_size, 'float32')
        values = np.zeros(self.step_size, 'float32')
        dones = np.zeros(self.step_size, 'int32')
        actions = np.array([action for _ in range(self.step_size)])
        prevactions = actions.copy()
        time_start = 0
        t_reset = 0
        round_reward = 0
        count_data = 0
        while True:
            prevaction = action
            while (time.time() - time_start < 0.05):
                pass

            action, value = self.act(ob)
            ob, reward, done = env.step(action[0])
            time_start = time.time()

            i = t % self.step_size

            obs[i] = ob
            values[i] = value
            dones[i] = done
            # print("actions is :")
            # print(action[0])
            actions[i] = action[0]
            prevactions[i] = prevaction
            
            ###env.render() #Show Animation -------------------------
            
            rewards[i] = reward
            cur_ep_return += reward
            
            cur_ep_length += 1
            if t > 0 and t_reset >= int((self.step_size/self.n_round)-1.0):
                count_data += int((self.step_size/self.n_round)-1.0)
                done = True



            if done:
                round_reward += cur_ep_return
                print("Reward: {}".format(cur_ep_return))
                ep_returns.append(cur_ep_return)
                ep_lengths.append(cur_ep_length)
                cur_ep_return = 0
                cur_ep_length = 0
                ###ob = env.reset()


                if t > 0 and count_data >= self.step_size:
                    print("Average Reward: {}".format(round_reward / self.n_round))
                    round_reward = 0.0
                    count_data = 0.0
                    yield {"ob": obs, "reward": rewards, "value": values,
                           "done": dones, "action": actions, "prevaction": prevactions,
                           "nextvalue": value * (1 - done), "ep_returns": ep_returns,
                           "ep_lengths": ep_lengths}

                env.end()
                env.setUp()
                t_reset = 0
                ep_returns = []
                ep_lengths = []
            t += 1
            t_reset += 1
        env.end()

    def act(self, ob):
        ### action, value = tfc.get_default_session().run([self.net.act_op, self.net.v], feed_dict={
        ###     self.net.obs_place: ob[None]
        ### })
        action, paction, value = tfc.get_default_session().run([self.net.act_op,self.net.p, self.net.v], feed_dict={
            self.net.obs_place: ob[None]
        })

        # print("&&&&&&&&&&&&&&&&&&")
        # print(action)
        # print(action[0])
        # print(value)
        # print("ACTION IS")
        # print(action)

        return action, value

    def run(self):
        traj_gen = self.traj_generator()
        iteration = 0

        while(True): #for _ in range(50): #-----200------  >>>>>>True

            iteration += 1
            # if(iteration!=1):
            #     env.end()
            #     env.setUp()
            print("\n================= iteration {} =================".format(iteration))
            traj = traj_gen.__next__()
            self.add_vtarg_and_adv(traj)

            tfc.get_default_session().run(self.assign_op)

            traj["advantage"] = (traj["advantage"] - np.mean(traj["advantage"])) / np.std(traj["advantage"])

            len = int((self.step_size-self.n_round) / self.batch_size)
            for _ in range(self.epochs):
                vf_loss = 0
                pol_loss = 0
                entropy = 0
                for i in range(len):
                    cur = i * self.batch_size + self.n_round
                    *step_losses, _ = tfc.get_default_session().run(
                        [self.ent, self.vf_loss, self.pol_loss, self.update_op],
                        feed_dict={self.obs_place: traj["ob"][cur:cur + self.batch_size],
                                   self.acts_place: traj["action"][cur:cur + self.batch_size],
                                   self.adv_place: traj["advantage"][cur:cur + self.batch_size],
                                   self.return_place: traj["return"][cur:cur + self.batch_size]})

                    entropy += step_losses[0] / len
                    vf_loss += step_losses[1] / len
                    pol_loss += step_losses[2] / len
                print("vf_loss: {:.5f}, pol_loss: {:.5f}, entorpy: {:.5f}".format(vf_loss, pol_loss, entropy))
            env.end()
            env.setUp()
            if iteration % 2 == 0:
                self.save_model('./model/checkpointPhoenixVREP_2.ckpt')
        env.end()


    def update(self):
        ent = self.entropy(self.net)
        ratio = tf.exp(self.logp(self.net) - tf.stop_gradient(self.logp(self.old_net)))
        surr1 = ratio * self.adv_place
        surr2 = tf.clip_by_value(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * self.adv_place

        pol_surr = -tf.reduce_mean(tf.minimum(surr1, surr2))
        vf_loss = tf.reduce_mean(tf.square(self.net.v - self.return_place))

        total_loss = pol_surr + 10 * vf_loss - 0.1*ent

        update_op = tfc.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(total_loss)

        return ent, pol_surr, vf_loss, update_op

    def add_vtarg_and_adv(self, traj):
        done = np.append(traj["done"], 0)
        value = np.append(traj["value"], traj["nextvalue"])
        T = len(traj["reward"])
        traj["advantage"] = gaelam = np.empty(T, 'float32')
        reward = traj["reward"]
        lastgaelam = 0

        for t in reversed(range(T)):
            nonterminal = 1 - done[t + 1]
            delta = reward[t] + self.gamma * value[t + 1] * nonterminal - value[t]
            gaelam[t] = lastgaelam = delta + self.gamma * self.lam * nonterminal * lastgaelam
        traj["return"] = traj["advantage"] + traj["value"]

    def save_model(self, model_path):
        self.saver.save(tfc.get_default_session(), model_path)
        print("model saved")

    def restore_model(self, model_path):
        self.saver.restore(tfc.get_default_session(), model_path)
        print("model restored")


if __name__ == "__main__":
    ###env = gym.make("MountainCarContinuous-v0")
    env = VREP_env()
    sess = tfc.InteractiveSession()
    ppo = PPOAgent(env)
    tfc.get_default_session().run(tfc.global_variables_initializer())
    ppo.restore_model("./model/checkpointPhoenixVREP_2.ckpt") #--------
    ppo.run()

    ###env.close()
    env.end