
import tensorflow as tf
from utility.helper import *
import numpy as np
from scipy.sparse import csr_matrix
from utility.batch_test import *
import os
import sys
import copy
import pickle

class WMF:
    def __init__(self, user_num, item_num,max_item_pu):
        self.user_num = user_num
        self.item_num = item_num
        self.embedding_size = args.embed_size
        self.max_item_pu = max_item_pu
        self.weight1 = args.negative_weight
        self.lambda_bilinear = [0, 0]
        self.lr = args.lr
        self.Ks = eval(args.Ks)

    def _create_placeholders(self):
        self.input_u = tf.placeholder(tf.int32, [None, 1], name="input_uid")
        self.input_ur = tf.placeholder(tf.int32, [None, None], name="input_ur")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        self.users = tf.placeholder(tf.int32, shape=(None,))
        self.pos_items = tf.placeholder(tf.int32, shape=(None,))

    def _create_variables(self):
        self.uidW = tf.Variable(tf.truncated_normal(shape=[self.user_num, self.embedding_size], mean=0.0,
                                                    stddev=0.01), dtype=tf.float32, name="uidWg")
        self.iidW = tf.Variable(tf.truncated_normal(shape=[self.item_num + 1, self.embedding_size], mean=0.0,
                                                    stddev=0.01), dtype=tf.float32, name="iidW")

    def _create_inference(self):
        self.uid = tf.nn.embedding_lookup(self.uidW, self.input_u)
        self.uid = tf.reshape(self.uid, [-1, self.embedding_size])

        self.uid = tf.nn.dropout(self.uid, self.dropout_keep_prob)

        self.pos_item = tf.nn.embedding_lookup(self.iidW, self.input_ur)
        self.pos_num_r = tf.cast(tf.not_equal(self.input_ur, self.item_num), 'float32')
        self.pos_item = tf.einsum('ab,abc->abc', self.pos_num_r, self.pos_item)
        self.pos_r = tf.einsum('ac,abc->ab', self.uid, self.pos_item)

    def _pre(self):
        u_e = tf.nn.embedding_lookup(self.uidW, self.users)
        pos_i_e = tf.nn.embedding_lookup(self.iidW, self.pos_items)
        self.batch_ratings = tf.matmul(u_e, pos_i_e, transpose_a=False, transpose_b=True)
    def _create_loss(self):
        self.loss1 = self.weight1 * tf.reduce_sum(tf.einsum('ab,ac->bc', self.iidW, self.iidW)
                          * tf.einsum('ab,ac->bc', self.uid, self.uid))
        self.loss1 += tf.reduce_sum((1.0 - self.weight1) * tf.square(self.pos_r) - 2.0 * self.pos_r)
        self.l2_loss0 = tf.nn.l2_loss(self.uidW)
        self.l2_loss1 = tf.nn.l2_loss(self.iidW)
        self.loss = self.loss1 \
                    + self.lambda_bilinear[0] * self.l2_loss0 \
                    + self.lambda_bilinear[1] * self.l2_loss1

        self.reg_loss = self.lambda_bilinear[0] * self.l2_loss0 \
                        + self.lambda_bilinear[1] * self.l2_loss1

        self.opt = tf.train.AdagradOptimizer(learning_rate=self.lr, initial_accumulator_value=1e-8).minimize(self.loss)

    def _build_graph(self):
        self._create_placeholders()
        self._create_variables()
        self._create_inference()
        self._create_loss()
        self._pre()
        #self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
        


def get_lables(temp_set,k=0.9999):
    max_item = 0
    item_lenth = []
    for i in temp_set:
        item_lenth.append(len(temp_set[i]))
        if len(temp_set[i]) > max_item:
            max_item = len(temp_set[i])
    item_lenth.sort()

    max_item = item_lenth[int(len(item_lenth) * k)-1]

    print max_item
    for i in temp_set:
        if len(temp_set[i]) > max_item:
            temp_set[i] = temp_set[i][0:max_item]
        while len(temp_set[i]) < max_item:
            temp_set[i].append(n_items)
    return max_item, temp_set

def get_train_instances(lable):
    user_train, item= [], []

    for i in lable.keys():
        user_train.append(i)
        item.append(lable[i])
    user_train = np.array(user_train)
    item = np.array(item)
    user_train = user_train[:, np.newaxis]
    return user_train, item

if __name__ == '__main__':

    tf.set_random_seed(2021)
    np.random.seed(2021)

    n_users, n_items = data_generator.n_users, data_generator.n_items
    train_items = copy.deepcopy(data_generator.train_items)
    max_item, lable = get_lables(train_items)

    t0 = time()

    model = WMF(n_users, n_items, max_item)
    model._build_graph()



    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    cur_best_pre_0 = 0.

    run_time = 1


    loss_loger, pre_loger, rec_loger, ndcg_loger,  = [], [], [], []

    stopping_step = 0
    should_stop = False

    user_train1, item1 = get_train_instances(lable)

    for epoch in range(args.epoch):
        shuffle_indices = np.random.permutation(np.arange(len(user_train1)))
        user_train1 = user_train1[shuffle_indices]
        item1 = item1[shuffle_indices]

        t1 = time()
        loss, mf_loss, emb_loss = 0., 0., 0.
        n_batch = int(len(user_train1) / args.batch_size)
        for idx in range(n_batch):
            start_index = idx * args.batch_size
            end_index = min((idx + 1) * args.batch_size, len(user_train1))

            u_batch = user_train1[start_index:end_index]
            v_batch = item1[start_index:end_index]
            _, batch_loss = sess.run(
                [model.opt, model.loss],
                feed_dict={model.input_u: u_batch,
                           model.input_ur: v_batch,
                           model.dropout_keep_prob:args.dropout
                           })
            loss += batch_loss / n_batch

        if np.isnan(loss) == True:
            print('ERROR: loss is nan.')
            sys.exit()

        if (epoch + 1) % 5 != 0:
            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f]' % (
                    epoch, time() - t1, loss)
                print(perf_str)
            continue

        t2 = time()
        users_to_test = list(data_generator.test_set.keys())
        ret = test(sess, model, users_to_test, drop_flag=False)

        t3 = time()
        loss_loger.append(loss)
        rec_loger.append(ret['recall'])
        pre_loger.append(ret['precision'])
        ndcg_loger.append(ret['ndcg'])

        if args.verbose > 0:
            perf_str = 'Epoch %d [%.1fs + %.1fs]:, recall=[%.5f, %.5f], ' \
                       'precision=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                       (epoch, t2 - t1, t3 - t2, ret['recall'][0], ret['recall'][1],
                        ret['precision'][0], ret['precision'][1],
                        ret['ndcg'][0], ret['ndcg'][1])
            print(perf_str)

        cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                    stopping_step, expected_order='acc', flag_step=10)
        if should_stop == True:
            break

    recs = np.array(rec_loger)
    pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)

    best_rec_0 = max(pres[:, 0])
    idx = list(pres[:, 0]).index(best_rec_0)

    final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], precision=[%s], ndcg=[%s]" % \
                 (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
                  '\t'.join(['%.5f' % r for r in pres[idx]]),
                  '\t'.join(['%.5f' % r for r in ndcgs[idx]]))
    print(final_perf)

    uidW, iidW = sess.run([model.uidW, model.iidW])

    with open(args.data_path + args.dataset + '/user_pretrain.pk', 'w') as f:
        pickle.dump(uidW, f)
    with open(args.data_path + args.dataset + '/item_pretrain.pk', 'w') as f:
        pickle.dump(iidW, f)
















