# sampler.py
import numpy as np
import random

def random_neq(l, r, s):
    """Sample ngẫu nhiên trong [l, r) sao cho không thuộc tập s."""
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

class WarpSampler(object):
    """
    Sampler tối giản, tương thích với code gốc:
      next_batch() -> (u, seq, pos, neg) với shape (B, maxlen)
      close() -> no-op
    """
    def __init__(self, User, usernum, itemnum, batch_size=128, maxlen=50, n_workers=1):
        self.User = User
        self.usernum = usernum
        self.itemnum = itemnum
        self.batch_size = batch_size
        self.maxlen = maxlen

    def _sample_for_user(self, u):
        """Sinh 1 (seq, pos, neg) cho user u theo đúng logic SASRec."""
        seq = np.zeros([self.maxlen], dtype=np.int32)
        pos = np.zeros([self.maxlen], dtype=np.int32)
        neg = np.zeros([self.maxlen], dtype=np.int32)

        hist = self.User[u]     # lịch sử items
        ts = set(hist)          # để lấy negative
        nxt = hist[-1]
        idx = self.maxlen - 1
        # duyệt history từ cuối lên (bỏ item cuối vì dùng làm nxt ban đầu)
        for i in reversed(hist[:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0:
                neg[idx] = random_neq(1, self.itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1:
                break
        return seq, pos, neg

    def next_batch(self):
        """Trả về một batch (u, seq, pos, neg)."""
        batch_u, batch_seq, batch_pos, batch_neg = [], [], [], []
        for _ in range(self.batch_size):
            u = np.random.randint(1, self.usernum + 1)
            # cần >=2 tương tác để tạo (seq,pos)
            while len(self.User.get(u, [])) < 2:
                u = np.random.randint(1, self.usernum + 1)
            seq, pos, neg = self._sample_for_user(u)
            batch_u.append(u)
            batch_seq.append(seq)
            batch_pos.append(pos)
            batch_neg.append(neg)
        return (np.array(batch_u, dtype=np.int32),
                np.array(batch_seq, dtype=np.int32),
                np.array(batch_pos, dtype=np.int32),
                np.array(batch_neg, dtype=np.int32))

    def close(self):
        pass
