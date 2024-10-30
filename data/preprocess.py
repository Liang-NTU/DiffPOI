import time
import argparse
import numpy as np
import pickle
from collections import Counter
from datetime import datetime

class DataFoursquare(object):
    def __init__(self, trace_min=10, global_visit=10, hour_gap=72, min_gap=10, session_min=2, session_max=10,
                 sessions_min=2, embedding_len=50, split_time=""):
        tmp_path = "./raw_data/"
        city_name = "NYC"
        self.INPUT_PATH = tmp_path + 'dataset_TSMC2014_' + city_name + '.txt'
        self.SAVE_PATH = tmp_path
        self.save_name = city_name

        self.trace_len_min = trace_min
        self.location_global_visit_min = global_visit
        self.hour_gap = hour_gap
        self.min_gap = min_gap
        self.session_max = session_max
        self.filter_short_session = session_min
        self.sessions_count_min = sessions_min
        self.words_embeddings_len = embedding_len

        self.split_time = split_time

        self.data = {}
        self.venues = {}
        self.rawpoi2feat = {}

        self.user2id = {} 
        self.poi2id = {} 
        self.pid2feat = {}

        self.all_pois = set()
        self.warm_pois = set()
        self.cold_pois = set()

        self.data_neural = {}

    # ############# 1. read trajectory data
    def load_trajectory(self):
        min_time = float("inf")
        max_time = 0

        with open(self.INPUT_PATH, "r", encoding='latin-1') as fid:
            for i, line in enumerate(fid):
                uid, pid, cid, cname, lat, lon, _, tim = line.strip().split("\t")
                
                if uid not in self.data:
                    self.data[uid] = [[pid, tim]]
                else:
                    self.data[uid].append([pid, tim])

                if pid not in self.venues:
                    self.venues[pid] = 1
                else:
                    self.venues[pid] += 1

                if pid not in self.rawpoi2feat:
                    self.rawpoi2feat[pid] = [cname, lat, lon]


                tid = int(time.mktime(time.strptime(tim, "%a %b %d %H:%M:%S +0000 %Y")))
                min_time = min(min_time, tid)
                max_time = max(max_time, tid)

        min_time = datetime.fromtimestamp(min_time).strftime("%Y-%m-%d")
        max_time = datetime.fromtimestamp(max_time).strftime("%Y-%m-%d")
        print("############## Checkin time range:", min_time, max_time)

    # ########### 2. basically filter users and pois
    def filter_user_poi(self):
        f_uids = [x for x in self.data if len(self.data[x]) > self.trace_len_min]
        f_uids = set(f_uids)

        f_pids = [x for x in self.venues if self.venues[x] > self.location_global_visit_min]
        f_pids = set(f_pids)

        self.f_uids = f_uids
        self.f_pids = f_pids

    # ########## 3. split training and testing based on time 
    def time_split(self):
        training = {}
        testing = {}

        split_time_stamp = datetime.strptime(self.split_time, '%Y-%m-%d').timestamp()

        for u in self.f_uids:
            user_data = self.data[u]

            if u not in training:
                training[u] = []
            if u not in testing:
                testing[u] = []
            
            for checkin in user_data:
                poi = checkin[0]
                tim = checkin[1]    

                if poi not in self.f_pids:
                    continue

                tid = int(time.mktime(time.strptime(tim, "%a %b %d %H:%M:%S +0000 %Y")))
                if tid < split_time_stamp:
                    training[u].append([poi, tid])
                else:
                    testing[u].append([poi, tid])

        # filter users without training data
        wt_filter_u = set()
        for u in training:
            user_data = training[u]
            if len(user_data) == 0:
                wt_filter_u.add(u)
        print("############## without training users:", len(wt_filter_u))

        # new training and testing
        f_training = {}
        f_testing = {}
        for u in training:
            if u in wt_filter_u:
                continue
            user_data = training[u]
            user_data = sorted(user_data, key=lambda x: x[1]) # sort is important
            f_training[u] = user_data
        for u in testing:
            if u in wt_filter_u:
                continue 
            user_data = testing[u]
            user_data = sorted(user_data, key=lambda x: x[1]) # sort is important
            f_testing[u] = user_data

        # build dictionary for users and location
        for u in f_training:
            user_data = f_training[u]
            if u not in self.user2id:
                self.user2id[u] = len(self.user2id)

            for checkin in user_data:
                poi = checkin[0]
                if poi not in self.poi2id:
                    self.poi2id[poi] = len(self.poi2id)

        for u in f_testing:
            user_data = f_testing[u]
            if u not in self.user2id:
                self.user2id[u] = len(self.user2id)

            for checkin in user_data:
                poi = checkin[0]
                if poi not in self.poi2id:
                    self.poi2id[poi] = len(self.poi2id)

        for p in self.poi2id:
            _pid = self.poi2id[p]
            self.pid2feat[_pid] = self.rawpoi2feat[p]

        # # refine training testing ids
        self.training = {}
        self.testing = {}
        for u in f_training:
            uid = self.user2id[u]
            self.training[uid] = []

            user_data = f_training[u]
            for checkin in user_data:
                poi = checkin[0]
                tid = checkin[1]
                pid = self.poi2id[poi]
                self.training[uid].append([pid, tid])
        for u in f_testing:
            uid = self.user2id[u]
            self.testing[uid] = []

            user_data = f_testing[u]
            for checkin in user_data:
                poi = checkin[0]
                tid = checkin[1]
                pid = self.poi2id[poi]
                self.testing[uid].append([pid, tid])

        for u in self.training:
            user_data = self.training[u]
            for checkin in user_data:
                poi = checkin[0]
                tid = checkin[1]
                self.warm_pois.add(poi)
                self.all_pois.add(poi)
        for u in self.testing:
            user_data = self.testing[u]
            for checkin in user_data:
                poi = checkin[0]
                tid = checkin[1]
                if poi not in self.warm_pois:
                    self.cold_pois.add(poi)
                self.all_pois.add(poi)
        print("############## all pois:", len(self.all_pois), "warm pois:", len(self.warm_pois), "cold pois:", len(self.cold_pois), "poi2id num:", len(self.poi2id))

    # ########## 4. prepare training data for neural network
    @staticmethod
    def tid_list(tmd):
        tm = time.strptime(tmd, "%a %b %d %H:%M:%S %z %Y")
        tid = tm.tm_wday * 24 + tm.tm_hour
        return tid

    @staticmethod
    def tid_list_48(tmd):
        # tm = time.strptime(tmd, "%a %b %d %H:%M:%S %z %Y")
        # if tm.tm_wday in [0, 1, 2, 3, 4]:
        #     tid = tm.tm_hour
        # else:
        #     tid = tm.tm_hour + 24
        # return tid

        tm = time.localtime(tmd)
        if tm.tm_wday in [0, 1, 2, 3, 4]:
            tid = tm.tm_hour
        else:
            tid = tm.tm_hour + 24
        return tid

    def trans2session(self, trans):
        sessions = {}
        for i, record in enumerate(trans):
            poi, tid = record
        
            sid = len(sessions)
            if i == 0 or len(sessions) == 0:
                sessions[sid] = [record]
            else:
                if (tid - last_tid) / 3600 > self.hour_gap or len(sessions[sid - 1]) > self.session_max:
                    sessions[sid] = [record]
                elif (tid - last_tid) / 60 > self.min_gap:
                    sessions[sid - 1].append(record)
                else:
                    sessions[sid - 1].append(record)
            last_tid = tid
        return sessions

    def prepare_neural_data(self):
        userset = set()
        poiset = set()
        for u in self.training:
            sessions = {}
            train_id = []
            test_id = []

            user_data = self.training[u]
            train_sessions = self.trans2session(user_data)
            for session_id in train_sessions:
                sessions[session_id] = train_sessions[session_id]
                train_id.append(session_id)

            if u in self.testing:
                user_data = self.testing[u]
                test_sessions = self.trans2session(user_data)
                for session_id in test_sessions:
                    new_session_id = session_id + len(train_sessions)
                    sessions[new_session_id] = test_sessions[session_id]
                    test_id.append(new_session_id)

            # sessions_tran = {}
            # for sid in sessions:
            #     sessions_tran[sid] = [[p[0], self.tid_list_48(p[1])] for p in
            #                           sessions[sid]]
            #     for p in sessions[sid]:
            #         poiset.add(p[0])

            sessions_tran = {}
            for sid in sessions:
                sessions_tran[sid] = [[p[0], p[1], float(self.pid2feat[p[0]][1]), float(self.pid2feat[p[0]][2])] for p in sessions[sid]]
                for p in sessions[sid]:
                    poiset.add(p[0])

            self.data_neural[u] = {'sessions': sessions_tran, 'train': train_id, 'test': test_id}
            userset.add(u)

    # ############# 5. save variables
    def get_parameters(self):
        parameters = {}
        parameters['INPUT_PATH'] = self.INPUT_PATH
        parameters['SAVE_PATH'] = self.SAVE_PATH

        parameters['trace_len_min'] = self.trace_len_min
        parameters['location_global_visit_min'] = self.location_global_visit_min
        parameters['hour_gap'] = self.hour_gap
        parameters['min_gap'] = self.min_gap
        parameters['session_max'] = self.session_max
        parameters['filter_short_session'] = self.filter_short_session
        parameters['sessions_min'] = self.sessions_count_min
        parameters['split_time'] = self.split_time

        return parameters

    def save_variables(self):
        foursquare_dataset = {'data_neural': self.data_neural, 'poi2id': self.poi2id, 'user2id': self.user2id,
                              'parameters': self.get_parameters(), 'pid2feat': self.pid2feat, 'all_pois': self.all_pois,
                              'warm_pois': self.warm_pois, 'cold_pois': self.cold_pois}
        pickle.dump(foursquare_dataset, open(self.SAVE_PATH + self.save_name + '.pk', 'wb'))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace_min', type=int, default=10, help="raw trace length filter threshold")
    parser.add_argument('--global_visit', type=int, default=5, help="location global visit threshold")
    parser.add_argument('--hour_gap', type=int, default=72, help="maximum interval of two trajectory points")
    parser.add_argument('--min_gap', type=int, default=10, help="minimum interval of two trajectory points")
    parser.add_argument('--session_max', type=int, default=10, help="control the length of session not too long")
    parser.add_argument('--session_min', type=int, default=5, help="control the length of session not too short")
    parser.add_argument('--sessions_min', type=int, default=5, help="the minimum amount of the good user's sessions")
    parser.add_argument('--split_time', type=str, default="2012-06-01", help="train/test split time")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    data_generator = DataFoursquare(trace_min=args.trace_min, global_visit=args.global_visit,
                                    hour_gap=args.hour_gap, min_gap=args.min_gap,
                                    session_min=args.session_min, session_max=args.session_max,
                                    sessions_min=args.sessions_min, split_time=args.split_time)
    parameters = data_generator.get_parameters()
    print('############PARAMETER SETTINGS:\n' + '\n'.join([p + ':' + str(parameters[p]) for p in parameters]))
    print('############START PROCESSING:')

    print('load trajectory from {}'.format(data_generator.INPUT_PATH))
    data_generator.load_trajectory()

    print('filter users & pois')
    data_generator.filter_user_poi()

    print('split training and testing based on time ')
    data_generator.time_split()

    print('prepare data for neural network')
    data_generator.prepare_neural_data()

    print('save prepared data')
    data_generator.save_variables()
    print('raw users:{} raw locations:{}'.format(
        len(data_generator.data), len(data_generator.venues)))
    print('final users:{} final locations:{}'.format(
        len(data_generator.data_neural), len(data_generator.pid2feat)))