import sys
import tensorflow as tf
from utility.helper import *
from utility.batch_test import *


class RecEraser_LightGCN(object):
    def __init__(self, data_config):
        # argument settings
        self.model_type = 'RecEraser_LightGCN'
        self.adj_type = args.adj_type
        self.alg_type = args.alg_type

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_fold = 3
        self.norm_adj = data_config['norm_adj']
        self.lr = args.lr
        self.emb_dim = args.embed_size

        self.attention_size = args.embed_size / 2

        self.batch_size = args.batch_size
        self.weight_size = eval(args.layer_size)
        self.n_layers = len(self.weight_size)
        self.regs = eval(args.regs)
        self.decay = self.regs[0]
        self.verbose = args.verbose
        self.Ks = eval(args.Ks)
        self.num_local = args.part_num

        self.weights = self._init_weights()

        self.users = tf.placeholder(tf.int32, shape=(None,))
        self.pos_items = tf.placeholder(tf.int32, shape=(None,))
        self.neg_items = tf.placeholder(tf.int32, shape=(None,))

        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        self.node_dropout_flag = args.node_dropout_flag
        self.node_dropout = tf.placeholder(tf.float32, shape=[None])
        self.mess_dropout = tf.placeholder(tf.float32, shape=[None])

        self.opt_local = []
        self.mf_loss_local = []
        self.reg_loss_local = []
        self.loss_local = []
        self.batch_ratings_local = []

        for i in range(self.num_local):
            print i
            line = self.train_single_model(i)
            self.opt_local.append(line[0])
            self.loss_local.append(line[1])
            self.mf_loss_local.append(line[2])
            self.reg_loss_local.append(line[3])
            self.batch_ratings_local.append(line[4])

        line = self.train_agg_model2()
        self.opt_agg = line[0]
        self.loss_agg = line[1]
        self.mf_loss_agg = line[2]
        self.batch_ratings = line[3]
        self.u_w = line[4]

    def train_single_model(self, local_num):

        ua_embeddings, ia_embeddings = self._create_lightgcn_embed_local(local_num)

        u_g_embeddings = tf.nn.embedding_lookup(ua_embeddings, self.users)
        pos_i_g_embeddings = tf.nn.embedding_lookup(ia_embeddings, self.pos_items)
        neg_i_g_embeddings = tf.nn.embedding_lookup(ia_embeddings, self.neg_items)

        u_g_embeddings_pre = tf.nn.embedding_lookup(self.weights['user_embedding'][:, local_num], self.users)
        pos_i_g_embeddings_pre = tf.nn.embedding_lookup(self.weights['item_embedding'][:, local_num], self.pos_items)
        neg_i_g_embeddings_pre = tf.nn.embedding_lookup(self.weights['item_embedding'][:, local_num], self.neg_items)

        mf_loss = self.create_bpr_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)

        regularizer = tf.nn.l2_loss(u_g_embeddings_pre) + tf.nn.l2_loss(
            pos_i_g_embeddings_pre) + tf.nn.l2_loss(neg_i_g_embeddings_pre)

        regularizer = regularizer / self.batch_size

        emb_loss = self.decay * regularizer

        loss = mf_loss + emb_loss

        batch_ratings = tf.matmul(u_g_embeddings, pos_i_g_embeddings, transpose_a=False,
                                  transpose_b=True)

        opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss)

        return opt, loss, mf_loss, emb_loss, batch_ratings

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
        ua_embeddings, ia_embeddings = [], []

        for i in range(self.num_local):
            ua_local, ia_local = self._create_lightgcn_embed_local(i)
            ua_embeddings.append(ua_local)
            ia_embeddings.append(ia_local)

        ua_embeddings = tf.transpose(ua_embeddings, [1, 0, 2])
        ia_embeddings = tf.transpose(ia_embeddings, [1, 0, 2])

        #u_g_embeddings = tf.stop_gradient(tf.nn.embedding_lookup(ua_embeddings, self.users))
        #pos_i_g_embeddings = tf.stop_gradient(tf.nn.embedding_lookup(ia_embeddings, self.pos_items))
        #neg_i_g_embeddings = tf.stop_gradient(tf.nn.embedding_lookup(ia_embeddings, self.neg_items))

        u_g_embeddings = tf.nn.embedding_lookup(ua_embeddings, self.users)
        pos_i_g_embeddings = tf.nn.embedding_lookup(ia_embeddings, self.pos_items)
        neg_i_g_embeddings = tf.nn.embedding_lookup(ia_embeddings, self.neg_items)

        u_es = tf.einsum('abc,bcd->abd', u_g_embeddings, self.weights['trans_W']) + self.weights['trans_B']
        pos_i_es = tf.einsum('abc,bcd->abd', pos_i_g_embeddings, self.weights['trans_W']) + self.weights['trans_B']
        neg_i_es = tf.einsum('abc,bcd->abd', neg_i_g_embeddings, self.weights['trans_W']) + self.weights['trans_B']

        u_e, u_w = self.attention_based_agg(u_es, 0)
        pos_i_e, pos_i_w = self.attention_based_agg(pos_i_es, 1)
        neg_i_e, neg_i_w = self.attention_based_agg(neg_i_es, 1)

        u_e_drop = tf.nn.dropout(u_e, self.dropout_keep_prob)

        mf_loss = self.create_bpr_loss(u_e_drop, pos_i_e, neg_i_e)

        loss = mf_loss

        batch_ratings = tf.matmul(u_e, pos_i_e, transpose_a=False, transpose_b=True)

        opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss)
        

        return opt, loss, mf_loss, batch_ratings, u_w

    def _create_lightgcn_embed(self,u_e,i_e):
        if self.node_dropout_flag:
            A_fold_hat = self._split_A_hat_node_dropout(self.norm_adj[self.num_local])
        else:
            A_fold_hat = self._split_A_hat(self.norm_adj[self.num_local])

        ego_embeddings = tf.concat([u_e, i_e], axis=0)
        all_embeddings = [ego_embeddings]

        for k in range(0, self.n_layers):

            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))

            side_embeddings = tf.concat(temp_embed, 0)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = tf.stack(all_embeddings, 1)
        all_embeddings = tf.reduce_mean(all_embeddings, axis=1, keepdims=False)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        return u_g_embeddings, i_g_embeddings

    def train_agg_model2(self):

        user_local_embs = tf.stop_gradient(self.weights['user_embedding'])
        item_local_embs = tf.stop_gradient(self.weights['item_embedding'])

        u_es = tf.einsum('abc,bcd->abd', user_local_embs, self.weights['trans_W'])+self.weights['trans_B']
        i_es = tf.einsum('abc,bcd->abd', item_local_embs, self.weights['trans_W'])+self.weights['trans_B']

        u_e, u_w = self.attention_based_agg(u_es, 0)
        i_e, i_w = self.attention_based_agg(i_es, 1)

        ###

        ua_embeddings, ia_embeddings = self._create_lightgcn_embed(u_e,i_e)

        u_g_embeddings = tf.nn.embedding_lookup(ua_embeddings, self.users)
        pos_i_g_embeddings = tf.nn.embedding_lookup(ia_embeddings, self.pos_items)
        neg_i_g_embeddings = tf.nn.embedding_lookup(ia_embeddings, self.neg_items)

        u_drop = tf.nn.dropout(u_g_embeddings, self.dropout_keep_prob)

        mf_loss = self.create_bpr_loss(u_drop, pos_i_g_embeddings, neg_i_g_embeddings)

        loss = mf_loss

        batch_ratings = tf.matmul(u_g_embeddings, pos_i_g_embeddings, transpose_a=False, transpose_b=True)

        opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss)
        

        return opt, loss, mf_loss, batch_ratings, u_w

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

    def _split_A_hat(self, X):
        A_fold_hat = []

        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len

            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat

    def _split_A_hat_node_dropout(self, X):
        A_fold_hat = []

        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len

            temp = self._convert_sp_mat_to_sp_tensor(X[start:end])
            n_nonzero_temp = X[start:end].count_nonzero()
            A_fold_hat.append(self._dropout_sparse(temp, 1 - self.node_dropout[0], n_nonzero_temp))

        return A_fold_hat

    def _create_lightgcn_embed_local(self, local_num):
        # print self.norm_adj[local_num]
        if self.node_dropout_flag:
            A_fold_hat = self._split_A_hat_node_dropout(self.norm_adj[local_num])
        else:
            A_fold_hat = self._split_A_hat(self.norm_adj[local_num])

        ego_embeddings = tf.concat(
            [self.weights['user_embedding'][:, local_num], self.weights['item_embedding'][:, local_num]], axis=0)
        all_embeddings = [ego_embeddings]

        for k in range(0, self.n_layers):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))

            side_embeddings = tf.concat(temp_embed, 0)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = tf.stack(all_embeddings, 1)
        all_embeddings = tf.reduce_mean(all_embeddings, axis=1, keepdims=False)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        return u_g_embeddings, i_g_embeddings

    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)

        mf_loss = tf.reduce_mean(tf.nn.softplus(-(pos_scores - neg_scores)))

        return mf_loss

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()

        return tf.SparseTensor(indices, coo.data, coo.shape)

    def _dropout_sparse(self, X, keep_prob, n_nonzero_elems):
        """
        Dropout for sparse tensors.
        """
        noise_shape = [n_nonzero_elems]
        random_tensor = keep_prob
        random_tensor += tf.random_uniform(noise_shape)
        dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
        pre_out = tf.sparse_retain(X, dropout_mask)

        return pre_out * tf.div(1., keep_prob)


if __name__ == '__main__':

    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items
    t0 = time()

    pre_adj = []
    for i in range(args.part_num):
        pre_adj.append(data_generator.get_adj_mat_local(i))

    pre_adj.append(data_generator.get_adj_mat())

    config['norm_adj'] = pre_adj
    print('use the pre adjcency matrix')

    model = RecEraser_LightGCN(data_config=config)

    saver = tf.train.Saver()
    # *********************************************************
    # save the model parameters.
    if args.save_flag == 1:
        weights_save_path = '%sweights/%s/%s/num-%s_type-%s_r%s+graph' % (
            args.proj_path, args.dataset, model.model_type, str(args.part_num), str(args.part_type),
            '-'.join([str(r) for r in eval(args.regs)]))
        ensureDir(weights_save_path)
        save_saver = tf.train.Saver(max_to_keep=1)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # reload the pretrained model parameters.
    if args.pretrain == 1:
        pretrain_path = '%sweights/%s/%s/num-%s_type-%s_r%s+graph' % (
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
            for epoch in range(args.epoch):
                t1 = time()
                loss, mf_loss, reg_loss = 0., 0., 0.
                n_batch = data_generator.n_C[i] // args.batch_size + 1

                for idx in range(n_batch):
                    users, pos_items, neg_items = data_generator.local_sample(i)

                    _, batch_loss, batch_mf_loss, batch_reg_loss = sess.run(
                        [model.opt_local[i], model.loss_local[i], model.mf_loss_local[i], model.reg_loss_local[i]],
                        feed_dict={model.users: users, model.pos_items: pos_items,
                                   model.neg_items: neg_items})
                    loss += batch_loss/ n_batch
                    mf_loss += batch_mf_loss/ n_batch
                    reg_loss += batch_reg_loss/ n_batch

                if np.isnan(loss) == True:
                    print('ERROR: loss is nan.')
                    sys.exit()

                if (epoch + 1) % 10 != 0:
                    if args.verbose > 0 and epoch % args.verbose == 0:
                        perf_str = '[local_model %d] Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
                            i, epoch, time() - t1, loss, mf_loss, reg_loss)
                        print(perf_str)
                    continue

                t2 = time()
                users_to_test = list(data_generator.test_set.keys())
                ret = test(sess, model, users_to_test, drop_flag=False, local_flag=True, local_num=i)

                t3 = time()

                if args.verbose > 0:
                    perf_str = '[local_model %d] Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f], recall=[%.5f, %.5f], ' \
                               'precision=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                               (i, epoch, t2 - t1, t3 - t2, loss, mf_loss, reg_loss, ret['recall'][0], ret['recall'][1],
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
    for epoch in range(args.epoch_agg):
        t1 = time()
        loss, mf_loss, reg_loss, attention_loss = 0., 0., 0., 0.
        n_batch = data_generator.n_train // args.batch_size + 1

        for idx in range(n_batch):
            # btime= time()
            users, pos_items, neg_items = data_generator.sample()
            _, batch_loss, batch_mf_loss, u_w = sess.run([model.opt_agg, model.loss_agg, model.mf_loss_agg, model.u_w],
                                                         feed_dict={model.users: users, model.pos_items: pos_items,
                                                                    model.neg_items: neg_items,
                                                                    model.dropout_keep_prob: args.dropout})
            loss += batch_loss
            mf_loss += batch_mf_loss

            # print(time() - btime)

        print u_w[0]

        if np.isnan(loss) == True:
            print('ERROR: loss is nan.')
            sys.exit()

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




















