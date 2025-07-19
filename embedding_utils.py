import numpy as np
import sqlite3
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
import joblib
import os
from transformers import AutoTokenizer, AutoModel
import torch

DB_PATH = 'database/roommate.db'
EMBEDDING_PATH = 'user_embeddings.npy'
ID_PATH = 'user_ids.npy'
VEC_PATH = 'vectorizer.pkl'
SCL_PATH = 'scaler.pkl'

# 切换为兼容性更好的bert-base-chinese
# 兼容性极好，分词器和模型都稳定

tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
model = AutoModel.from_pretrained('bert-base-chinese')
model.eval()

def get_text_embedding(text):
    if not text or not isinstance(text, str):
        text = ""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=32, padding="max_length")
    with torch.no_grad():
        outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    return cls_embedding

# 获取新注册用户的全部字段（只查最新一位）
def fetch_latest_user():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, gender, major, sleep_time, wake_time, ac_temp, cleanliness, bunk_preference, study_habits, weekend_stay, group_life, hobbies, mbti FROM user ORDER BY id DESC LIMIT 1")
    row = cursor.fetchone()
    conn.close()
    return row

# 只对新用户做embedding，追加到缓存
def add_new_user_embedding():
    user_row = fetch_latest_user()
    if user_row is None:
        return
    user_dict = {
        'gender': user_row[1],
        'major': user_row[2],
        'sleep_time': user_row[3],
        'wake_time': user_row[4],
        'ac_temp': user_row[5],
        'cleanliness': user_row[6],
        'bunk_preference': user_row[7],
        'study_habits': user_row[8],
        'weekend_stay': user_row[9],
        'group_life': user_row[10],
    }
    hobbies = user_row[11]
    # 加载已有
    if os.path.exists(EMBEDDING_PATH) and os.path.exists(ID_PATH) and os.path.exists(VEC_PATH) and os.path.exists(SCL_PATH):
        X = np.load(EMBEDDING_PATH)
        user_ids = np.load(ID_PATH)
        vec = joblib.load(VEC_PATH)
        scl = joblib.load(SCL_PATH)
        X_new = vec.transform([user_dict])
        X_scaled = scl.transform(X_new)
    else:
        vec = DictVectorizer(sparse=False)
        X_new = vec.fit_transform([user_dict])
        scl = StandardScaler()
        X_scaled = scl.fit_transform(X_new)
        X = np.empty((0, X_scaled.shape[1] + 768))  # 768是bert-base-chinese输出维度
        user_ids = np.array([], dtype=np.int64)
        joblib.dump(vec, VEC_PATH)
        joblib.dump(scl, SCL_PATH)
    # 结构化特征加权
    feature_names = vec.get_feature_names_out()
    feature_weights = {
        'sleep_time': 5,
        'wake_time': 5,
        'cleanliness': 3,
        'study_habits': 2,
        'bunk_preference': 2,
    }
    X_weighted = X_scaled.copy()
    for i, name in enumerate(feature_names):
        for key, w in feature_weights.items():
            if key in name:
                X_weighted[:, i] *= w
    bert_weight = 0.5
    hobbies_emb = get_text_embedding(hobbies)
    final_emb = np.concatenate([X_weighted[0], bert_weight * hobbies_emb])
    if X.shape[0] == 0:
        X_all = np.expand_dims(final_emb, axis=0)
        user_ids_all = np.array([user_row[0]])
    else:
        X_all = np.vstack([X, final_emb])
        user_ids_all = np.append(user_ids, user_row[0])
    np.save(EMBEDDING_PATH, X_all)
    np.save(ID_PATH, user_ids_all)

# 新用户注册后调用
update_embeddings_if_new_user = add_new_user_embedding

# 修正版：匹配时排除自己，只匹配已通过审核且未被封禁的用户
def find_top3_matches(current_user_dict, current_hobbies, current_user_id, same_major_only=False, current_major=None):
    if not (os.path.exists(EMBEDDING_PATH) and os.path.exists(ID_PATH) and os.path.exists(VEC_PATH) and os.path.exists(SCL_PATH)):
        return []
    X = np.load(EMBEDDING_PATH)
    user_ids = np.load(ID_PATH)
    vec = joblib.load(VEC_PATH)
    scl = joblib.load(SCL_PATH)
    current_vec = vec.transform([current_user_dict])
    current_vec_scaled = scl.transform(current_vec)
    # 结构化特征加权
    feature_names = vec.get_feature_names_out()
    feature_weights = {
        'sleep_time': 5,
        'wake_time': 5,
        'cleanliness': 3,
        'study_habits': 2,
        'bunk_preference': 2,
    }
    current_vec_weighted = current_vec_scaled.copy()
    for i, name in enumerate(feature_names):
        for key, w in feature_weights.items():
            if key in name:
                current_vec_weighted[:, i] *= w
    bert_weight = 0.5
    current_hobbies_emb = get_text_embedding(current_hobbies)
    current_full_emb = np.concatenate([current_vec_weighted[0], bert_weight * current_hobbies_emb])
    # 过滤掉自己和不合格用户
    from models import User
    valid_mask = []
    user_bunks = []
    for uid in user_ids:
        user = User.query.get(int(uid))
        if user and user.is_approved and not user.is_banned and int(uid) != int(current_user_id):
            if same_major_only and current_major and user.major != current_major:
                valid_mask.append(False)
                user_bunks.append(None)
            else:
                valid_mask.append(True)
                user_bunks.append(user.bunk_preference)
        else:
            valid_mask.append(False)
            user_bunks.append(None)
    X_valid = X[valid_mask]
    user_ids_valid = user_ids[valid_mask]
    user_bunks_valid = [b for b, v in zip(user_bunks, valid_mask) if v]
    if len(user_ids_valid) == 0:
        return []
    # 计算距离，床铺偏好特殊处理
    def bunk_distance(b1, b2):
        # 上铺优先匹配下铺或无偏好
        if (b1 == '上铺' and (b2 == '下铺' or b2 == '无')) or (b2 == '上铺' and (b1 == '下铺' or b1 == '无')):
            return 0
        elif b1 == b2:
            return 1
        else:
            return 2
    bunk_weight = 2
    dists = []
    for i, emb in enumerate(X_valid):
        # 结构化+爱好距离
        dist = np.linalg.norm(current_full_emb - emb)
        # 床铺距离
        bunk_dist = bunk_distance(current_user_dict.get('bunk_preference'), user_bunks_valid[i])
        dists.append(dist + bunk_weight * bunk_dist)
    top3_idx = np.argsort(dists)[:3]
    return user_ids_valid[top3_idx].tolist() 