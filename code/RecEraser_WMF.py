import tensorflow as tf
from utility.helper import *
from utility.batch_test import *
import os
import sys
import copy
import pickle

class RecEraser_WMF(object):
    def __init__(self, user_num, item_num,C_I):
        self.model_type = 'RecEraser_WMF'
        self.n_users = user_num
        self.n_items = item_num
        self.emb_dim = args.embed_size
        self.attention_size = args.embed_size / 2
        self.weight1 = args.negative_weight
        self.lambda_bilinear = [0, 0]
        self.lr = args.lr
        self.Ks = eval(args.Ks)
        self.num_local = args.part_num
        self.C_I = C_I

        self.input_ur = tf.placeholder(tf.int32, [None, None], name="input_ur")
        self.dropout_keep_prob_local = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep")

        self.users = tf.placeholder(tf.int32, shape=(None,))
        self.pos_items = tf.placeholder(tf.int32, shape=(None,))
        self.weights = self._init_weights()

        self.opt_local = []
        self.loss_local = []
        self.batch_ratings_local = []

        for i in range(self.num_local):
            line = self.train_single_model(i)
            self.opt_local.append(line[0])
            self.loss_local.append(line[1])
            self.batch_ratings_local.append(line[2])

        line = self.train_agg_model()
        self.opt_agg = line[0]
        self.loss_agg = line[1]
        self.batch_ratings = line[2]
        self.u_w = line[3]

    def _init_weights(self):
        all_weights = dict()
        initializer = tf.random_normal_initializer(stddev=0.01)  # tf.contrib.layers.xavier_initializer()

        all_weights['user_embedding'] = tf.Variable(initializer([self.n_users, self.num_local, self.emb_dim]),
                                                    name='user_embedding')
        all_weights['item_embedding'] = tf.Variable(initializer([self.n_items, self.num_local, self.emb_dim]),
                                                    name='item_embedding')
        # user attention
        all_weights['WA'] = tf.Variable(
            tf.truncated_normal(shape=[self.emb_dim, self.attention_size], mean=0.0, stddev=tf.sqrt(
                tf.div(2.0, self.attention_size + self.emb_dim))), dtype=tf.float32, name='WA')
        all_weights['BA'] = tf.Variable(tf.constant(0.00, shape=[self.attention_size]), name="BA")
        all_weights['HA'] = tf.Variable(tf.constant(0.01, shape=[self.attention_size, 1]), name="HA")

        # item attention
        all_weights['WB'] = tf.Variable(
            tf.truncated_normal(shape=[self.emb_dim, self.attention_size], mean=0.0, stddev=tf.sqrt(
                tf.div(2.0, self.attention_size + self.emb_dim))), dtype=tf.float32, name='WB')
        all_weights['BB'] = tf.Variable(tf.constant(0.00, shape=[self.attention_size]), name="BB")
        all_weights['HB'] = tf.Variable(tf.constant(0.01, shape=[self.attention_size, 1]), name="HB")

        # trans weights

        all_weights['trans_W'] = tf.Variable(initializer([self.num_local, self.emb_dim, self.emb_dim]),
                                             name='user_embedding')
        all_weights['trans_B'] = tf.Variable(initializer([self.num_local, self.emb_dim]), name='user_embedding')

        return all_weights

    def train_single_model(self, local_num):
        u_e = tf.nn.embedding_lookup(self.weights['user_embedding'][:, local_num], self.users)
        u_e = tf.reshape(u_e, [-1, self.emb_dim])
        u_e_drop = tf.nn.dropout(u_e, self.dropout_keep_prob_local)

        pos_item = tf.nn.embedding_lookup(self.weights['item_embedding'][:, local_num], self.input_ur)
        pos_num_r = tf.cast(tf.not_equal(self.input_ur, self.n_items), 'float32')
        pos_item = tf.einsum('ab,abc->abc', pos_num_r, pos_item)
        pos_r = tf.einsum('ac,abc->ab', u_e_drop, pos_item)

        local_iw = tf.nn.embedding_lookup(self.weights['item_embedding'][:, local_num], self.C_I[local_num])
        local_iw = tf.reshape(local_iw, [-1, self.emb_dim])

        loss = self._create_loss(u_e_drop,local_iw,pos_r)
        opt = tf.train.AdagradOptimizer(learning_rate=self.lr, initial_accumulator_value=1e-8).minimize(loss)


        u_e = u_e = tf.nn.embedding_lookup(self.weights['user_embedding'][:, local_num], self.users)
        pos_i_e = tf.nn.embedding_lookup(self.weights['item_embedding'][:, local_num], self.pos_items)
        batch_ratings = tf.matmul(u_e, pos_i_e, transpose_a=False, transpose_b=True)

        return opt, loss, batch_ratings

    def attention_based_agg(self, embs, flag):
        if flag == 0:
            embs_w = tf.exp(
                tf.einsum('abc,ck->abk', tf.nn.relu(
                    tf.einsum('abc,ck->abk', embs, self.weights['WA']) + self.weights['BA']),
                          self.weights['HA']))

            embs_w = tf.div(embs_w, tf.reduce_sum(embs_w, 1, keep_dims=True))
        else:
            embs_w = tf.exp(
                tf.einsum('abc,ck->abk', tf.nn.relu(
                    tf.einsum('abc,ck->abk', embs, self.weights['WB']) + self.weights['BB']),
                          self.weights['HB']))

            embs_w = tf.div(embs_w, tf.reduce_sum(embs_w, 1, keep_dims=True))

        agg_emb = tf.reduce_sum(tf.multiply(embs_w, embs), 1)

        return agg_emb, embs_w

    def train_agg_model(self):
        u_es = tf.stop_gradient(tf.nn.embedding_lookup(self.weights['user_embedding'], self.users))
        u_es = tf.reshape(u_es, [-1,self.num_local, self.emb_dim])
        u_es = tf.einsum('abc,bcd->abd', u_es, self.weights['trans_W']) + self.weights['trans_B']
        u_e, u_w = self.attention_based_agg(u_es, 0)
        u_e_drop = tf.nn.dropout(u_e, self.dropout_keep_prob)

        item_local_embs = tf.stop_gradient(self.weights['item_embedding'])
        i_es = tf.einsum('abc,bcd->abd', item_local_embs, self.weights['trans_W']) + self.weights['trans_B']
        i_e, i_w = self.attention_based_agg(i_es, 1)

        pos_item = tf.nn.embedding_lookup(i_e, self.input_ur)
        pos_num_r = tf.cast(tf.not_equal(self.input_ur, self.n_items), 'float32')
        pos_item = tf.einsum('ab,abc->abc', pos_num_r, pos_item)
        pos_r = tf.einsum('ac,abc->ab', u_e_drop, pos_item)

        loss = self._create_loss(u_e_drop, i_e, pos_r)
        opt = tf.train.AdagradOptimizer(learning_rate=self.lr, initial_accumulator_value=1e-8).minimize(loss)

        pos_i_e = tf.nn.embedding_lookup(i_e, self.pos_items)
        batch_ratings = tf.matmul(u_e, pos_i_e, transpose_a=False, transpose_b=True)

        return opt, loss, batch_ratings, u_w


    def _create_loss(self,user,items,pos_r):
        loss = self.weight1 * tf.reduce_sum(tf.einsum('ab,ac->bc', items, items)
                                                  * tf.einsum('ab,ac->bc', user, user))
        loss += tf.reduce_sum((1.0 - self.weight1) * tf.square(pos_r) - 2.0 * pos_r)
        return loss

def get_lables(temp_set, k=0.9999):
    max_item = 0
    item_lenth = []
    for i in temp_set:
        item_lenth.append(len(temp_set[i]))
        if len(temp_set[i]) > max_item:
            max_item = len(temp_set[i])
    item_lenth.sort()

    max_item = item_lenth[int(len(item_lenth) * k) - 1]
    for i in temp_set:
        if len(temp_set[i]) > max_item:
            temp_set[i] = temp_set[i][0:max_item]
        while len(temp_set[i]) < max_item:
            temp_set[i].append(n_items)
    return temp_set

def get_train_instances(lable):
    user_train, item= [], []

    for i in lable.keys():
        user_train.append(i)
        item.append(lable[i])
    user_train = np.array(user_train)
    item = np.array(item)
    #user_train = user_train[:, np.newaxis]
    return user_train, item

if __name__ == '__main__':
    tf.set_random_seed(2021)
    np.random.seed(2021)

    n_users, n_items = data_generator.n_users, data_generator.n_items
    train_items = copy.deepcopy(data_generator.train_items)

    local_train_items = copy.deepcopy(data_generator.C)
    local_users = copy.deepcopy(data_generator.C_U)
    local_items = copy.deepcopy(data_generator.C_I)
    train_lable = get_lables(train_items)

    local_train_lable=[]
    for i in range(args.part_num):
        local_train_lable.append(get_lables(local_train_items[i]))

    t0 = time()
    model = RecEraser_WMF(n_users, n_items,local_items)

    saver = tf.train.Saver()
    # *********************************************************
    # save the model parameters.
    if args.save_flag == 1:
        weights_save_path = '%sweights/%s/%s/num-%s_type-%s_r%s' % (
            args.proj_path, args.dataset, model.model_type, str(args.part_num), str(args.part_type),
            '-'.join([str(r) for r in eval(args.regs)]))
        ensureDir(weights_save_path)
        save_saver = tf.train.Saver(max_to_keep=1)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # reload the pretrained model parameters.

    if args.pretrain == 1:
        pretrain_path = '%sweights/%s/%s/num-%s_type-%s_r%s' % (
            args.proj_path, args.dataset, model.model_type, str(args.part_num), str(args.part_type),
            '-'.join([str(r) for r in eval(args.regs)]))

        ckpt = tf.train.get_checkpoint_state(os.path.dirname(pretrain_path + '/checkpoint'))
        print ckpt
        if ckpt and ckpt.model_checkpoint_path:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('load the pretrained model parameters from: ', pretrain_path)

    else:
        sess.run(tf.global_variables_initializer())

        for i in range(args.part_num):
            cur_best_pre_0 = 0.
            stopping_step = 0
            user_train1, item1 = get_train_instances(local_train_lable[i])
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
                        [model.opt_local[i], model.loss_local[i]],
                        feed_dict={model.users: u_batch,
                                   model.input_ur: v_batch,
                                   model.dropout_keep_prob_local: args.dropout
                                   })
                    loss += batch_loss / n_batch

                if np.isnan(loss) == True:
                    print('ERROR: loss is nan.')
                    sys.exit()
                if (epoch + 1) % 10 != 0:
                    if args.verbose > 0 and epoch % args.verbose == 0:
                        perf_str = '[local_model %d] Epoch %d [%.1fs]: train==[%.5f]' % (i,
                            epoch, time() - t1, loss)
                        print(perf_str)
                    continue

                t2 = time()
                users_to_test = list(data_generator.test_set.keys())
                ret = test(sess, model, users_to_test, drop_flag=False, local_flag=True, local_num=i)

                t3 = time()
                if args.verbose > 0:
                    perf_str = '[local_model %d]  Epoch %d [%.1fs + %.1fs]:, recall=[%.5f, %.5f], ' \
                               'precision=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                               (i,epoch, t2 - t1, t3 - t2, ret['recall'][0], ret['recall'][1],
                                ret['precision'][0], ret['precision'][1],
                                ret['ndcg'][0], ret['ndcg'][1])
                    print(perf_str)

                cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                            stopping_step, expected_order='acc',
                                                                            flag_step=10)
                if should_stop == True:
                    break

        save_saver.save(sess, weights_save_path + '/weights')
        print('save the weights in path: ', weights_save_path)
    # train agg
    cur_best_pre_0 = 0.
    stopping_step = 0

    user_train1, item1 = get_train_instances(train_lable)
    for epoch in range(args.epoch_agg):
        shuffle_indices = np.random.permutation(np.arange(len(user_train1)))
        user_train1 = user_train1[shuffle_indices]
        item1 = item1[shuffle_indices]
        t1 = time()
        loss, mf_loss, reg_loss, attention_loss = 0., 0., 0., 0.
        n_batch = int(len(user_train1) / args.batch_size)
        for idx in range(n_batch):
            start_index = idx * args.batch_size
            end_index = min((idx + 1) * args.batch_size, len(user_train1))

            u_batch = user_train1[start_index:end_index]
            v_batch = item1[start_index:end_index]

            _, batch_loss,u_w  = sess.run(
                [model.opt_agg, model.loss_agg,model.u_w],
                feed_dict={model.users: u_batch,
                           model.input_ur: v_batch,
                           model.dropout_keep_prob: args.dropout
                           })
            loss += batch_loss / n_batch

        print u_w[0]
        if (epoch + 1) % 1 != 0:
            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f ]' % (epoch, time() - t1, loss, mf_loss)
                print(perf_str)
            continue

        t2 = time()
        users_to_test = list(data_generator.test_set.keys())
        ret = test(sess, model, users_to_test, drop_flag=False)
        t3 = time()

        if args.verbose > 0:
            perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f ], recall=[%.5f, %.5f, %.5f], ' \
                       'precision=[%.5f, %.5f,%.5f], ndcg=[%.5f, %.5f,%.5f]' % \
                       (epoch, t2 - t1, t3 - t2, loss, mf_loss, ret['recall'][0], ret['recall'][1], ret['recall'][2],
                        ret['precision'][0], ret['precision'][1], ret['precision'][2],
                        ret['ndcg'][0], ret['ndcg'][1], ret['ndcg'][2])
            print(perf_str)

        cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                    stopping_step, expected_order='acc', flag_step=10)
        if should_stop == True:
            break





































