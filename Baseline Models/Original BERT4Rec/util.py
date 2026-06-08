import sys
import copy
import random
import numpy as np
from collections import defaultdict
import math

def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    f = open(fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)
    f.close()

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = [User[user][-2]]
            user_test[user] = [User[user][-1]]
    return [user_train, user_valid, user_test, usernum, itemnum]

def evaluate(model, dataset, args, sess):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    # Khởi tạo ks giống BERT4Rec
    ks = [5, 10, 15]
    metrics = {f'NDCG@{k}': 0.0 for k in ks}
    metrics.update({f'HR@{k}': 0.0 for k in ks})
    
    valid_user = 0.0
    np.random.seed(42)

    users = random.sample(range(1, usernum + 1), 10000) if usernum > 10000 else range(1, usernum + 1)
    
    for u in users:
        train_u = train.get(u, [])
        valid_u = valid.get(u, [])
        test_u  = test.get(u, [])

        if len(train_u) < 1 or len(valid_u) < 1 or len(test_u) < 1:
            continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid_u[0]
        idx -= 1
        for i in reversed(train_u):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train_u) | set(valid_u) | set(test_u) | {0}

        # Tạo candidate list: 1 positive + 100 negatives (giống BERT của bạn)
        item_idx = [test_u[0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated:
                t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        # Lấy dự đoán từ model
        predictions = model.predict(sess, [u], [seq], item_idx)[0]
        
        # LOGIC GIỐNG BERT: argsort giảm dần để lấy rank
        # Trong BERT: rank = (-scores).argsort(dim=1)
        rank = (-predictions).argsort() 
        
        # labels giống BERT: [1, 0, 0, ...] vì item đúng nằm ở index 0 của item_idx
        labels = np.zeros(len(item_idx))
        labels[0] = 1 

        valid_user += 1

        # Tính toán theo công thức của BERT4Rec bạn gửi
        for k in ks:
            cut = rank[:k]
            hits = labels[cut] # Những index nào trong Top-K là item đúng

            # HitRate
            if np.sum(hits) > 0:
                metrics[f'HR@{k}'] += 1

            # NDCG logic: dcg / idcg
            # position tương ứng: 2, 3, ..., k+1
            dcg = 0.0
            for i in range(len(hits)):
                if hits[i] > 0:
                    dcg += 1.0 / math.log2(i + 2)
            
            # IDCG cho 1 item đúng luôn là 1/log2(2) = 1.0
            # Nếu answer_count > 1 thì idcg = sum(1/log2(i+2)) cho i từ 0 đến count-1
            idcg = 1.0 
            metrics[f'NDCG@{k}'] += (dcg / idcg)

        if valid_user % 100 == 0:
            print('.', end='')
            sys.stdout.flush()

    if valid_user == 0: return (0,0,0,0,0,0)

    return (
        metrics['NDCG@5'] / valid_user, metrics['HR@5'] / valid_user,
        metrics['NDCG@10'] / valid_user, metrics['HR@10'] / valid_user,
        metrics['NDCG@15'] / valid_user, metrics['HR@15'] / valid_user
    )

def evaluate_valid(model, dataset, args, sess):
    # Logic tương tự như evaluate nhưng dùng tập valid_u làm mục tiêu
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)
    ks = [5, 10, 15]
    metrics = {f'NDCG@{k}': 0.0 for k in ks}
    metrics.update({f'HR@{k}': 0.0 for k in ks})
    valid_user = 0.0
    np.random.seed(42)

    users = random.sample(range(1, usernum + 1), 10000) if usernum > 10000 else range(1, usernum + 1)
    for u in users:
        train_u = train.get(u, [])
        valid_u = valid.get(u, [])
        if len(train_u) < 1 or len(valid_u) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train_u):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train_u) | set(valid_u) | {0}
        item_idx = [valid_u[0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = model.predict(sess, [u], [seq], item_idx)[0]
        rank = (-predictions).argsort()
        labels = np.zeros(len(item_idx))
        labels[0] = 1

        valid_user += 1
        for k in ks:
            cut = rank[:k]
            hits = labels[cut]
            if np.sum(hits) > 0: metrics[f'HR@{k}'] += 1
            dcg = sum([1.0 / math.log2(i + 2) for i in range(len(hits)) if hits[i] > 0])
            metrics[f'NDCG@{k}'] += dcg # idcg = 1.0

    if valid_user == 0: return (0,0,0,0,0,0)
    return (
        metrics['NDCG@5'] / valid_user, metrics['HR@5'] / valid_user,
        metrics['NDCG@10'] / valid_user, metrics['HR@10'] / valid_user,
        metrics['NDCG@15'] / valid_user, metrics['HR@15'] / valid_user
    )