import torch
from torch.autograd import Variable

import numpy as np
# import cPickle as pickle
import pickle
from collections import deque, Counter
import time 

class RnnParameterData(object):
    def __init__(self, loc_emb_size=500, uid_emb_size=40, voc_emb_size=50, tim_emb_size=10, hidden_size=500,
                 lr=1e-3, lr_step=3, lr_decay=0.1, dropout_p=0.5, L2=1e-5, clip=5.0, optim='Adam',
                 history_mode='avg', attn_type='dot', epoch_max=30, rnn_type='LSTM', model_mode="simple",
                 data_path='../data/', save_path='../results/', data_name='NYC'):
        self.data_path = data_path
        self.save_path = save_path
        self.data_name = data_name
        data = pickle.load(open(self.data_path + self.data_name + '.pk', 'rb'), encoding='iso-8859-1')

        self.poi2id = data["poi2id"]
        self.user2id = data["user2id"]
        self.pid2feat = data["pid2feat"]
        self.all_pois = data["all_pois"]
        self.warm_pois = data["warm_pois"]
        self.cold_pois = data["cold_pois"]
        self.data_neural = data['data_neural']

        self.tim_size = 48
        self.loc_size = len(self.poi2id)
        self.uid_size = len(self.user2id)
        self.loc_emb_size = loc_emb_size
        self.tim_emb_size = tim_emb_size
        self.voc_emb_size = voc_emb_size
        self.uid_emb_size = uid_emb_size
        self.hidden_size = hidden_size

        self.epoch = epoch_max
        self.dropout_p = dropout_p
        self.use_cuda = True
        self.lr = lr
        self.lr_step = lr_step
        self.lr_decay = lr_decay
        self.optim = optim
        self.L2 = L2
        self.clip = clip

        self.attn_type = attn_type
        self.rnn_type = rnn_type
        self.history_mode = history_mode
        self.model_mode = model_mode

def generate_input_long_history(data_neural, mode, candidate=None):
    data_train = {}
    train_idx = {}
    if candidate is None:
        candidate = data_neural.keys()
    for u in candidate:
        sessions = data_neural[u]['sessions']
        train_id = data_neural[u][mode]
        data_train[u] = {}

        trace = {}
        session = []
        for c, i in enumerate(train_id):
            session.extend(sessions[i])
        target = np.array([s[0] for s in session[1:]])

        if len(target) == 0:
            continue

        loc_tim_coord = []
        loc_tim_coord.extend([(s[0], float(s[1]), [s[2], s[3]]) for s in session[:-1]])

        loc_np = np.reshape(np.array([s[0] for s in loc_tim_coord]), (len(loc_tim_coord), 1))
        tim_np = np.reshape(np.array([s[1] for s in loc_tim_coord]), (len(loc_tim_coord), 1))
        coord_np = np.reshape(np.array([s[2] for s in loc_tim_coord]), (len(loc_tim_coord), 2))

        trace['loc'] = Variable(torch.LongTensor(loc_np))

        # trace['tim'] = Variable(torch.LongTensor(tim_np))
        trace['tim'] = Variable(torch.FloatTensor(tim_np))
        trace['coord'] = Variable(torch.FloatTensor(coord_np))

        trace['target'] = Variable(torch.LongTensor(target))
        data_train[u][i] = trace

        if mode == 'train':
            train_idx[u] = [0, i]
        else:
            train_idx[u] = [i]
    return data_train, train_idx


def generate_queue(train_idx, mode, mode2):
    """return a deque. You must use it by train_queue.popleft()"""
    user = list(train_idx.keys())
    train_queue = deque()
    if mode == 'random':
        initial_queue = {}
        for u in user:
            if mode2 == 'train':
                initial_queue[u] = deque(train_idx[u][1:])
            else:
                initial_queue[u] = deque(train_idx[u])
        queue_left = 1
        while queue_left > 0:
            np.random.shuffle(user)
            for j, u in enumerate(user):
                if len(initial_queue[u]) > 0:
                    train_queue.append((u, initial_queue[u].popleft()))
                if j >= int(0.01 * len(user)):
                    break
            queue_left = sum([1 for x in initial_queue if len(initial_queue[x]) > 0])
    elif mode == 'normal':
        for u in user:
            for i in train_idx[u]:
                train_queue.append((u, i))
    return train_queue


def get_acc(target, scores):
    """target and scores are torch cuda Variable"""
    target = target.data.cpu().numpy()
    val, idxx = scores.data.topk(10, 1)
    predx = idxx.cpu().numpy()
    acc = np.zeros((3, 1))
    for i, p in enumerate(predx):
        t = target[i]
        if t in p[:3] and t > 0:
            acc[0] += 1
        if t in p[:5] and t > 0:
            acc[1] += 1
        if t in p[:10] and t > 0:
            acc[2] += 1
    return acc

def run_simple(data, run_idx, mode, lr, clip, model, optimizer, criterion, mode2=None, parameters=None):
    """mode=train: return model, avg_loss
       mode=test: return avg_loss,avg_acc,users_rnn_acc"""

    cold_pois = parameters.cold_pois
    warm_pois = parameters.warm_pois

    run_queue = None
    if mode == 'train':
        model.train(True)
        run_queue = generate_queue(run_idx, 'random', 'train')
    elif mode == 'test':
        model.train(False)
        run_queue = generate_queue(run_idx, 'normal', 'test')

    total_loss = []
    queue_len = len(run_queue)

    start = time.time()
    test_cases = [0, 0, 0]
    users_acc = {}
    case_acc = [0, 0, 0, 0, 0, 0]
    for c in range(queue_len):
        optimizer.zero_grad()
        u, i = run_queue.popleft()
        if u not in users_acc:
            users_acc[u] = [0, 0, 0, 0, 0, 0]
        loc = data[u][i]['loc'].cuda()
        tim = data[u][i]['tim'].cuda()
        coord = data[u][i]["coord"].cuda()
        target = data[u][i]['target'].cuda()
        uid = Variable(torch.LongTensor([u])).cuda()

        scores = model(loc, tim, coord)

        if scores.data.size()[0] > target.data.size()[0]:
            scores = scores[-target.data.size()[0]:]
        loss = criterion(scores, target)

        if mode == 'train':
            loss.backward()
            # gradient clipping
            try:
                torch.nn.utils.clip_grad_norm(model.parameters(), clip)
                for p in model.parameters():
                    if p.requires_grad:
                        p.data.add_(-lr, p.grad.data)
            except:
                pass
            optimizer.step()

        elif mode == 'test':
            users_acc[u][0] += len(target)
            acc = get_acc(target, scores)
            users_acc[u][1] += acc[2]
            case_acc[0] += len(target)
            case_acc[1] += acc[2]

            warm_target, warm_scores = [], []
            cold_target, cold_scores = [], []
            for i in range(len(target)):
                poi = target[i].item()
                if poi in warm_pois:
                    warm_target.append(target[i])
                    warm_scores.append(scores[i])
                else:
                    cold_target.append(target[i])
                    cold_scores.append(scores[i])
            
            if len(warm_target) > 0:
                warm_target = torch.stack(warm_target)
                warm_scores = torch.stack(warm_scores)
                users_acc[u][2] += len(warm_target)
                acc = get_acc(warm_target, warm_scores)
                users_acc[u][3] += acc[2]
                case_acc[2] += len(warm_target)
                case_acc[3] += acc[2]

            if len(cold_target) > 0:
                cold_target = torch.stack(cold_target)
                cold_scores = torch.stack(cold_scores)
                users_acc[u][4] += len(cold_target)
                acc = get_acc(cold_target, cold_scores)
                users_acc[u][5] += acc[2]
                case_acc[4] += len(cold_target)
                case_acc[5] += acc[2]

            test_cases[0] += len(target)
            test_cases[1] += len(warm_target)
            test_cases[2] += len(cold_target)

        total_loss.append(loss.data.cpu().numpy())

    avg_loss = np.mean(total_loss, dtype=np.float64)
    if mode == 'train':
        return model, avg_loss
    elif mode == 'test':
        users_rnn_acc = {}
        for u in users_acc:
            tmp_acc = users_acc[u][1] / users_acc[u][0]
            users_rnn_acc[u] = tmp_acc.tolist()[0]

        warm_acc_list = []
        for u in users_acc:
            if users_acc[u][2] != 0:
                warm_acc = users_acc[u][3] / users_acc[u][2]
                warm_acc_list.append(warm_acc)
        avg_warm_acc = np.mean(warm_acc_list)

        cold_acc_list = []
        for u in users_acc:
            if users_acc[u][4] != 0:
                cold_acc = users_acc[u][5] / users_acc[u][4]
                cold_acc_list.append(cold_acc)
        avg_cold_acc = np.mean(cold_acc_list)

        avg_acc_case = case_acc[1][0] / case_acc[0]
        avg_warm_acc_case = case_acc[3][0] / case_acc[2]
        avg_cold_acc_case = case_acc[5][0] / case_acc[4]

        print('==>Test Cases:{:.1f} Warm Case:{:.1f} Cold Cases:{:.1f}'.format(test_cases[0], test_cases[1], test_cases[2]))

        avg_acc = np.mean([users_rnn_acc[x] for x in users_rnn_acc])
        return avg_loss, avg_acc, users_rnn_acc, avg_warm_acc, avg_cold_acc, avg_acc_case, avg_warm_acc_case, avg_cold_acc_case