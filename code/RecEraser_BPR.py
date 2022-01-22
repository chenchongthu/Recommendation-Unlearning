import tensorflow as tf
from utility.helper import *
import numpy as np
from scipy.sparse import csr_matrix
from utility.batch_test import *
import os
import sys
import pickle
import copy

import random

class RecEraser_BPR(object):
    def __init__(self, data_config):
        self.model_type = 'RecEraser_BPR'
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.lr = args.lr

        self.emb_dim = args.embed_size
        self.attention_size = args.embed_size/2
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

        self.opt_local =[] 
        self.mf_loss_local =[]
        self.reg_loss_local =[]
        self.loss_local = []
        self.batch_ratings_local=[]
        for i in range(self.num_local):
            line = self.train_single_model(i)
            self.opt_local.append(line[0])
            self.loss_local.append(line[1])
            self.mf_loss_local.append(line[2])
            self.reg_loss_local.append(line[3])
            self.batch_ratings_local.append(line[4])
            

        line = self.train_agg_model()
        self.opt_agg = line[0]
        self.loss_agg = line[1]
        self.mf_loss_agg = line[2]
        self.reg_loss_agg = line[3]
        self.attention_loss = line[4]
        self.batch_ratings = line[5]
        self.u_w = line[6]

    def _init_weights(self):
        all_weights = dict()

        initializer = tf.contrib.layers.xavier_initializer()

        all_weights['user_embedding'] = tf.Variable(initializer([self.n_users, self.num_local, self.emb_dim]), name='user_embedding')
        all_weights['item_embedding'] = tf.Variable(initializer([self.n_items, self.num_local, self.emb_dim]), name='item_embedding')

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

        all_weights['trans_W'] = tf.Variable(initializer([self.num_local, self.emb_dim, self.emb_dim]), name='user_embedding')
        all_weights['trans_B'] = tf.Variable(initializer([self.num_local, self.emb_dim]), name='user_embedding')

        return all_weights


    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)

        regularizer = tf.nn.l2_loss(users) + tf.nn.l2_loss(pos_items) + tf.nn.l2_loss(neg_items)
        regularizer = regularizer/self.batch_size

        maxi = tf.log(tf.nn.sigmoid(pos_scores - neg_scores))

        mf_loss = tf.negative(tf.reduce_mean(maxi))
        reg_loss = self.decay * regularizer
        return mf_loss, reg_loss


    def train_single_model(self, local_num):

        u_e = tf.nn.embedding_lookup(self.weights['user_embedding'][:,local_num], self.users)
        pos_i_e = tf.nn.embedding_lookup(self.weights['item_embedding'][:,local_num], self.pos_items)
        neg_i_e = tf.nn.embedding_lookup(self.weights['item_embedding'][:,local_num], self.neg_items)

        #print u_e,pos_i_e

        mf_loss, reg_loss = self.create_bpr_loss(u_e, pos_i_e, neg_i_e)
        loss = mf_loss + reg_loss

        batch_ratings = tf.matmul(u_e, pos_i_e, transpose_a=False, transpose_b=True)

        opt = tf.train.AdagradOptimizer(learning_rate=self.lr, initial_accumulator_value=1e-8).minimize(loss)

        return opt, loss, mf_loss, reg_loss, batch_ratings



    def attention_based_agg(self,embs,flag):

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

    def attention_based_agg2 (self,embs):


        embs_w = tf.exp(
            tf.einsum('abc,ck->abk', tf.nn.relu(
                tf.einsum('abc,ck->abk', embs, self.weights['WA']) + self.weights['BA']),
                      self.weights['HA']))

        embs_w = tf.div(embs_w, tf.reduce_sum(embs_w, 1, keep_dims=True))

        agg_emb = tf.reduce_sum(tf.multiply(embs_w, embs), 1)


        return agg_emb, embs_w


    def train_agg_model(self):

        u_es = tf.stop_gradient(tf.nn.embedding_lookup(self.weights['user_embedding'], self.users))
        pos_i_es = tf.stop_gradient(tf.nn.embedding_lookup(self.weights['item_embedding'], self.pos_items))
        neg_i_es = tf.stop_gradient(tf.nn.embedding_lookup(self.weights['item_embedding'], self.neg_items))

        '''embs1 = u_es*pos_i_es
        embs2 = u_es*neg_i_es

        dot1, u_w= self.attention_based_agg2(embs1)
        dot2,w = self.attention_based_agg2(embs2)

        l2_loss = 1e-4*(tf.nn.l2_loss(self.weights['WA']) + tf.nn.l2_loss(self.weights['BA']) + tf.nn.l2_loss(
            self.weights['HA']))

        loss = tf.negative(tf.reduce_mean(tf.log(tf.nn.sigmoid(tf.reduce_sum(dot1, axis=1) - tf.reduce_sum(dot2, axis=1)))))+l2_loss

        
        embs = tf.einsum('abc,ybc->aybc', u_es, pos_i_es)
        embs_w = tf.exp(
            tf.einsum('aybc,ck->aybk', tf.nn.relu(
                tf.einsum('aybc,ck->aybk', embs, self.weights['WA']) + self.weights['BA']),
                      self.weights['HA']))

        embs_w = tf.div(embs_w, tf.reduce_sum(embs_w, 2, keep_dims=True))

        agg_emb = tf.reduce_sum(tf.multiply(embs_w, embs), 2)
        batch_ratings = tf.reduce_sum(agg_emb,2)'''




        u_es = tf.einsum('abc,bcd->abd', u_es, self.weights['trans_W'])+self.weights['trans_B']
        pos_i_es = tf.einsum('abc,bcd->abd', pos_i_es, self.weights['trans_W'])+self.weights['trans_B']
        neg_i_es = tf.einsum('abc,bcd->abd', neg_i_es, self.weights['trans_W'])+self.weights['trans_B']

        u_e, u_w = self.attention_based_agg(u_es, 0)
        pos_i_e, pos_i_w = self.attention_based_agg(pos_i_es, 1)
        neg_i_e, neg_i_w = self.attention_based_agg(neg_i_es, 1)

        u_e_drop=tf.nn.dropout(u_e, self.dropout_keep_prob)

        #u_e = tf.reduce_mean(u_es,1)
        #pos_i_e = tf.reduce_mean(pos_i_es,1)
        #neg_i_e = tf.reduce_mean(neg_i_es,1)



        mf_loss, reg_loss = self.create_bpr_loss(u_e_drop, pos_i_e, neg_i_e)

        l2_loss = self.regs[0]*(tf.nn.l2_loss(self.weights['WA']) + tf.nn.l2_loss(self.weights['BA']) + tf.nn.l2_loss(
            self.weights['HA']) + tf.nn.l2_loss(self.weights['WB']) + tf.nn.l2_loss(self.weights['BB']) + tf.nn.l2_loss(
            self.weights['HB']))

        reg_loss = 1e-5*(tf.nn.l2_loss(self.weights['trans_W']) + tf.nn.l2_loss(self.weights['trans_B'])) 

        batch_ratings = tf.matmul(u_e, pos_i_e, transpose_a=False, transpose_b=True)

        loss = mf_loss + reg_loss # + l2_loss'''




        opt = tf.train.AdagradOptimizer(learning_rate=self.lr, initial_accumulator_value=1e-8).minimize(loss)

        return opt, loss, mf_loss, reg_loss, l2_loss, batch_ratings,u_w



if __name__ == '__main__':

    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items

    t0 = time()

    model = RecEraser_BPR(data_config=config)

    saver = tf.train.Saver()

    # *********************************************************
    # save the model parameters.
    if args.save_flag == 1:
        weights_save_path = '%sweights/%s/%s/num-%s_type-%s_r%s' % (args.proj_path, args.dataset, model.model_type, str(args.part_num),str(args.part_type),
                                                         '-'.join([str(r) for r in eval(args.regs)]))
        ensureDir(weights_save_path)
        save_saver = tf.train.Saver(max_to_keep=1)



    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)


    # reload the pretrained model parameters.
    if args.pretrain == 1:
        pretrain_path = '%sweights/%s/%s/num-%s_type-%s_r%s' % (args.proj_path, args.dataset, model.model_type, str(args.part_num),str(args.part_type),
                                                         '-'.join([str(r) for r in eval(args.regs)]))

        ckpt = tf.train.get_checkpoint_state(os.path.dirname(pretrain_path + '/checkpoint'))
        print ckpt
        if ckpt and ckpt.model_checkpoint_path:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('load the pretrained model parameters from: ', pretrain_path)

    else:
        sess.run(tf.global_variables_initializer())
        #train single model

        for i in range(args.part_num):
            cur_best_pre_0 = 0.
            stopping_step = 0
            for epoch in range(args.epoch):
                t1 = time()
                loss, mf_loss, reg_loss = 0., 0., 0.
                n_batch = data_generator.n_C[i] // args.batch_size + 1

                for idx in range(n_batch):
                    users, pos_items, neg_items = data_generator.local_sample(i)

                    _, batch_loss, batch_mf_loss, batch_reg_loss = sess.run([model.opt_local[i], model.loss_local[i], model.mf_loss_local[i], model.reg_loss_local[i]],
                                   feed_dict={model.users: users, model.pos_items: pos_items,
                                              model.neg_items: neg_items})
                    loss += batch_loss
                    mf_loss += batch_mf_loss
                    reg_loss += batch_reg_loss

                if np.isnan(loss) == True:
                    print('ERROR: loss is nan.')
                    sys.exit()

                if (epoch + 1) % 5 != 0:
                    if args.verbose > 0 and epoch % args.verbose == 0:
                        perf_str = '[local_model %d] Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (i, epoch, time()-t1, loss, mf_loss, reg_loss)
                        print(perf_str)
                    continue


                t2 = time()
                users_to_test = list(data_generator.test_set.keys())
                ret = test(sess, model, users_to_test, drop_flag=False, local_flag=True,local_num=i)

                t3 = time()

                if args.verbose > 0:
                    perf_str = '[local_model %d] Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f], recall=[%.5f, %.5f], ' \
                               'precision=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                               (i, epoch, t2 - t1, t3 - t2, loss, mf_loss, reg_loss, ret['recall'][0], ret['recall'][1],
                                ret['precision'][0], ret['precision'][1],
                                ret['ndcg'][0], ret['ndcg'][1])
                    print(perf_str)

                cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                        stopping_step, expected_order='acc', flag_step=10)
                if should_stop == True:
                    break

        save_saver.save(sess, weights_save_path + '/weights')
        print('save the weights in path: ', weights_save_path)



    #train agg
    cur_best_pre_0 = 0.
    stopping_step = 0
    for epoch in range(args.epoch_agg):
        t1 = time()
        loss, mf_loss, reg_loss,attention_loss = 0., 0., 0.,0.
        n_batch = data_generator.n_train // args.batch_size + 1

        for idx in range(n_batch):
            # btime= time()
            users, pos_items, neg_items = data_generator.sample()
            _, batch_loss, batch_mf_loss, batch_reg_loss, batch_attention_loss,u_w = sess.run([model.opt_agg, model.loss_agg, model.mf_loss_agg, model.reg_loss_agg,model.attention_loss,model.u_w],
                               feed_dict={model.users: users, model.pos_items: pos_items,
                                          model.neg_items: neg_items,
                                          model.dropout_keep_prob: args.dropout})
            loss += batch_loss
            mf_loss += batch_mf_loss
            reg_loss += batch_reg_loss
            attention_loss += batch_attention_loss
            # print(time() - btime)

        print u_w[0]

        if np.isnan(loss) == True:
            print('ERROR: loss is nan.')
            sys.exit()

        if (epoch + 1) % 1 != 0:
            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f+ %.5f]' % (epoch, time()-t1, loss, mf_loss, reg_loss,attention_loss)
                print(perf_str)
            continue

        t2 = time()
        users_to_test = list(data_generator.test_set.keys())
        ret = test(sess, model, users_to_test, drop_flag=False)

        t3 = time()


        if args.verbose > 0:
            perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f+ %.5f], recall=[%.5f, %.5f, %.5f], ' \
                       'precision=[%.5f, %.5f,%.5f], ndcg=[%.5f, %.5f,%.5f]' % \
                       (epoch, t2 - t1, t3 - t2, loss, mf_loss, reg_loss, attention_loss, ret['recall'][0], ret['recall'][1], ret['recall'][2],
                        ret['precision'][0], ret['precision'][1],ret['precision'][2],
                        ret['ndcg'][0], ret['ndcg'][1],ret['ndcg'][2])
            print(perf_str)

        cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                    stopping_step, expected_order='acc', flag_step=10)
        if should_stop == True:
            break




































