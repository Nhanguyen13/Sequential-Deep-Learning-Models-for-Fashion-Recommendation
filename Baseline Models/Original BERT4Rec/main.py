import os
import time
import argparse
import numpy as np

try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
except Exception:
    import tensorflow as tf

from sampler import WarpSampler
from model import Model
from tqdm import tqdm
from util import data_partition, evaluate, evaluate_valid

def str2bool(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--train_dir', required=True)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=40, type=int)
parser.add_argument('--hidden_units', default=256, type=int) 
parser.add_argument('--num_blocks', default=2, type=int)   
parser.add_argument('--num_epochs', default=50, type=int) 
parser.add_argument('--num_heads', default=2, type=int)    
parser.add_argument('--dropout_rate', default=0.5, type=float) 
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--patience', default=5, type=int)

args = parser.parse_args()

out_dir = args.dataset + '_' + args.train_dir
if not os.path.isdir(out_dir):
    os.makedirs(out_dir)

dataset = data_partition(args.dataset)
[user_train, user_valid, user_test, usernum, itemnum] = dataset
num_batch = len(user_train) // args.batch_size 

f_log = open(os.path.join(out_dir, 'log.txt'), 'w')
f_log.write(f"{'='*120}\n")
f_log.flush()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
sess = tf.Session(config=config)

sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
model = Model(usernum, itemnum, args)
sess.run(tf.global_variables_initializer())

t0 = time.time()

best_valid_ndcg = -1.0
best_epoch = 0
patience = args.patience 
patience_counter = 0

try:
    for epoch in range(1, args.num_epochs + 1):
        loss_list = []
        pbar = tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b', desc=f"Epoch {epoch}")
        for step in pbar:
            u, seq, pos, neg = sampler.next_batch()
            _, loss = sess.run([model.train_op, model.loss],
                               {model.u: u, 
                                model.input_seq: seq, 
                                model.pos: pos, 
                                model.neg: neg,
                                model.is_training: True})
            loss_list.append(loss)
            pbar.set_postfix(loss=f'{loss:.4f}')

        avg_loss = np.mean(loss_list)

        # LUÔN ĐÁNH GIÁ VALID ĐỂ KIỂM TRA EARLY STOPPING
        t_valid = evaluate_valid(model, dataset, args, sess)
        current_valid_ndcg = t_valid[2] 

        is_best = False
        if current_valid_ndcg > best_valid_ndcg:
            best_valid_ndcg = current_valid_ndcg
            best_epoch = epoch
            patience_counter = 0
            is_best = True
        else:
            patience_counter += 1

        # LOGGING THEO YÊU CẦU: Epoch 1, Mỗi 3 epoch, HOẶC khi đạt Best
        if epoch == 1 or epoch % 3 == 0 or is_best:
            t_test = evaluate(model, dataset, args, sess)
            
            # In đầy đủ các chỉ số bao gồm HR@10
            log_valid = (f"VALID | NDCG@5: {t_valid[0]:.4f} | HR@5: {t_valid[1]:.4f} | NDCG@10: {t_valid[2]:.4f} | HR@10: {t_valid[3]:.4f}")
            log_test =  (f"TEST  | NDCG@5: {t_test[0]:.4f} | HR@5: {t_test[1]:.4f} | NDCG@10: {t_test[2]:.4f} | HR@10: {t_test[3]:.4f}")

            status = "(NEW BEST!)" if is_best else ""
            print(f"\n[Epoch {epoch}/{args.num_epochs}] Loss: {avg_loss:.4f} {status}")
            print(log_valid)
            print(log_test)
            print(f"{'-' * 100}")
            
            f_log.write(f"\n[Epoch {epoch}] Loss: {avg_loss:.4f}\n{log_valid}\n{log_test}\n")
        else:
            # Các epoch bình thường in gọn
            print(f"Epoch {epoch}/{args.num_epochs} - Loss: {avg_loss:.4f}")
            f_log.write(f"Epoch {epoch} - Loss: {avg_loss:.4f}\n")
            
        f_log.flush()

        if patience_counter >= patience:
            msg = f"\n[Early Stopping] Dừng tại Epoch {epoch}. Best Epoch: {best_epoch}"
            print(msg)
            f_log.write(msg + "\n")
            break

except Exception as e:
    sampler.close()
    f_log.close()
    raise e

f_log.close()
sampler.close()
print(f"Training Done. Best Validation NDCG@10: {best_valid_ndcg:.4f} at Epoch {best_epoch}")