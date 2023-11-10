"""
Created on Oct 10, 2018
Tensorflow Implementation of the baseline of "Matrix Factorization with BPR Loss" in:
Wang Xiang et al. Neural Graph Collaborative Filtering. In SIGIR 2019.
@author: Xiang Wang (xiangwang@u.nus.edu)
"""
import sys
from time import time

import numpy as np
import tensorflow.compat.v1 as tf
from utility.batch_test import args, data_generator, test
from utility.helper import early_stopping

tf.disable_v2_behavior()


class BPRMF(object):
    def __init__(self, data_config):
        self.model_type = "bprmf"

        self.n_users = data_config["n_users"]
        self.n_items = data_config["n_items"]

        self.lr = args.lr
        # self.lr_decay = args.lr_decay

        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size

        self.weight_size = eval(args.layer_size)
        self.n_layers = len(self.weight_size)

        self.regs = eval(args.regs)
        self.decay = self.regs[0]

        self.verbose = args.verbose

        # placeholder definition
        self.users = tf.placeholder(tf.int32, shape=(None,))
        self.pos_items = tf.placeholder(tf.int32, shape=(None,))
        self.neg_items = tf.placeholder(tf.int32, shape=(None,))
        # self.global_step = tf.Variable(0, trainable=False)

        self.weights = self._init_weights()

        self.Ks = eval(args.Ks)

        # Original embedding.
        u_e = tf.nn.embedding_lookup(self.weights["user_embedding"], self.users)
        pos_i_e = tf.nn.embedding_lookup(self.weights["item_embedding"], self.pos_items)
        neg_i_e = tf.nn.embedding_lookup(self.weights["item_embedding"], self.neg_items)

        # All ratings for all users.
        self.batch_ratings = tf.matmul(u_e, pos_i_e, transpose_a=False, transpose_b=True)

        self.mf_loss, self.reg_loss = self.create_bpr_loss(u_e, pos_i_e, neg_i_e)
        self.loss = self.mf_loss + self.reg_loss

        # self.dy_lr = tf.train.exponential_decay(self.lr, self.global_step, 10000, self.lr_decay, staircase=True)
        # self.opt = tf.train.RMSPropOptimizer(learning_rate=self.lr).minimize(self.loss)
        self.opt = tf.train.AdagradOptimizer(
            learning_rate=self.lr, initial_accumulator_value=1e-8
        ).minimize(self.loss)
        # self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

        # self.updates = self.opt.minimize(self.loss, var_list=self.weights)

        self._statistics_params()

    def _init_weights(self):
        all_weights = dict()

        initializer = tf.initializers.glorot_normal()

        all_weights["user_embedding"] = tf.Variable(
            initializer([self.n_users, self.emb_dim]), name="user_embedding"
        )
        all_weights["item_embedding"] = tf.Variable(
            initializer([self.n_items, self.emb_dim]), name="item_embedding"
        )

        return all_weights

    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)

        regularizer = tf.nn.l2_loss(users) + tf.nn.l2_loss(pos_items) + tf.nn.l2_loss(neg_items)
        regularizer = regularizer / self.batch_size

        maxi = tf.log(tf.nn.sigmoid(pos_scores - neg_scores))

        mf_loss = tf.negative(tf.reduce_mean(maxi))
        reg_loss = self.decay * regularizer
        return mf_loss, reg_loss

    def _statistics_params(self):
        # number of params
        total_parameters = 0
        for variable in self.weights.values():
            shape = variable.get_shape()  # shape is an array of tf.Dimension
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        if self.verbose > 0:
            print("#params: %d" % total_parameters)


if __name__ == "__main__":
    config = dict()
    config["n_users"] = data_generator.n_users
    config["n_items"] = data_generator.n_items

    t0 = time()

    model = BPRMF(data_config=config)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    sess.run(tf.global_variables_initializer())
    cur_best_pre_0 = 0.0
    print("without pretraining.")

    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    stopping_step = 0

    for epoch in range(args.epoch):
        t1 = time()
        loss, mf_loss, reg_loss = 0.0, 0.0, 0.0
        n_batch = data_generator.n_train // args.batch_size + 1

        for idx in range(n_batch):
            # btime= time()
            users, pos_items, neg_items = data_generator.sample()
            _, batch_loss, batch_mf_loss, batch_reg_loss = sess.run(
                [model.opt, model.loss, model.mf_loss, model.reg_loss],
                feed_dict={
                    model.users: users,
                    model.pos_items: pos_items,
                    model.neg_items: neg_items,
                },
            )
            loss += batch_loss
            mf_loss += batch_mf_loss
            reg_loss += batch_reg_loss
            # print(time() - btime)

        if np.isnan(loss) is True:
            print("ERROR: loss is nan.")
            sys.exit()

        # print the test evaluation metrics each 10 epochs; pos:neg = 1:10.
        if (epoch + 1) % 5 != 0:
            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = "Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]" % (
                    epoch,
                    time() - t1,
                    loss,
                    mf_loss,
                    reg_loss,
                )
                print(perf_str)
            continue

        t2 = time()
        users_to_test = list(data_generator.test_set.keys())
        ret = test(sess, model, users_to_test, drop_flag=False)

        t3 = time()

        loss_loger.append(loss)
        rec_loger.append(ret["recall"])
        pre_loger.append(ret["precision"])
        ndcg_loger.append(ret["ndcg"])

        if args.verbose > 0:
            perf_str = (
                "Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f], recall=[%.5f, %.5f], "
                "precision=[%.5f, %.5f], ndcg=[%.5f, %.5f]"
                % (
                    epoch,
                    t2 - t1,
                    t3 - t2,
                    loss,
                    mf_loss,
                    reg_loss,
                    ret["recall"][0],
                    ret["recall"][1],
                    ret["precision"][0],
                    ret["precision"][1],
                    ret["ndcg"][0],
                    ret["ndcg"][1],
                )
            )
            print(perf_str)

        cur_best_pre_0, stopping_step, should_stop = early_stopping(
            ret["recall"][0],
            cur_best_pre_0,
            stopping_step,
            expected_order="acc",
            flag_step=5,
        )
        if should_stop is True:
            break

    recs = np.array(rec_loger)
    pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)
    hit = np.array(hit_loger)

    best_rec_0 = max(pres[:, 0])
    idx = list(pres[:, 0]).index(best_rec_0)

    final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], precision=[%s], ndcg=[%s]" % (
        idx,
        time() - t0,
        "\t".join(["%.5f" % r for r in recs[idx]]),
        "\t".join(["%.5f" % r for r in pres[idx]]),
        "\t".join(["%.5f" % r for r in ndcgs[idx]]),
    )
    print(final_perf)
