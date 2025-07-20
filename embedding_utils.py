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
    # 过滤掉自己和不合格用户，添加性别过滤
    from models import User
    valid_mask = []
    user_bunks = []
    current_gender = current_user_dict.get('gender')
    
    # 性别值映射
    gender_mapping = {'male': '男', 'female': '女', '男': '男', '女': '女'}
    current_gender_normalized = gender_mapping.get(current_gender, current_gender)
    
    for uid in user_ids:
        user = User.query.get(int(uid))
        if user and user.is_approved and not user.is_banned and int(uid) != int(current_user_id):
            # 添加性别过滤，只匹配同性别用户
            user_gender_normalized = gender_mapping.get(user.gender, user.gender)
            if user_gender_normalized != current_gender_normalized:
                valid_mask.append(False)
                user_bunks.append(None)
                continue
                
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
    
    # 获取用户详细信息用于睡眠时间计算
    user_details = []
    for uid in user_ids_valid:
        user = User.query.get(int(uid))
        if user:
            user_details.append({
                'sleep_time': user.sleep_time,
                'wake_time': user.wake_time,
                'bunk_preference': user.bunk_preference
            })
        else:
            user_details.append({
                'sleep_time': '',
                'wake_time': '',
                'bunk_preference': ''
            })
    
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
    
    def sleep_overlap_score(sleep1, wake1, sleep2, wake2):
        """计算睡眠时间重合度得分（线性计算）"""
        try:
            # 解析时间格式 "HH:MM"
            def time_to_minutes(time_str):
                if not time_str or ':' not in time_str:
                    return 0
                hours, minutes = map(int, time_str.split(':'))
                return hours * 60 + minutes
            
            # 转换为分钟
            sleep1_min = time_to_minutes(sleep1)
            wake1_min = time_to_minutes(wake1)
            sleep2_min = time_to_minutes(sleep2)
            wake2_min = time_to_minutes(wake2)
            
            if sleep1_min == 0 or wake1_min == 0 or sleep2_min == 0 or wake2_min == 0:
                return 0
            
            # 处理跨夜的情况
            if sleep1_min > wake1_min:  # 跨夜
                wake1_min += 24 * 60
            if sleep2_min > wake2_min:  # 跨夜
                wake2_min += 24 * 60
            
            # 计算重叠时间
            overlap_start = max(sleep1_min, sleep2_min)
            overlap_end = min(wake1_min, wake2_min)
            
            if overlap_end <= overlap_start:
                return 0  # 没有重叠
            
            overlap_minutes = overlap_end - overlap_start
            
            # 计算总睡眠时长（以最早睡和最晚起为准）
            earliest_sleep = min(sleep1_min, sleep2_min)
            latest_wake = max(wake1_min, wake2_min)
            total_sleep_duration = latest_wake - earliest_sleep
            
            # 计算重合度：重合时间占总睡眠时长的比例
            overlap_percentage = overlap_minutes / total_sleep_duration if total_sleep_duration > 0 else 0
            
            return overlap_percentage
            
        except Exception as e:
            print(f"睡眠时间计算错误: {e}")
            return 0
    
    bunk_weight = 1
    sleep_weight = 5  # 睡眠重合度权重（最高优先级）
    dists = []
    for i, emb in enumerate(X_valid):
        # 结构化+爱好距离
        dist = np.linalg.norm(current_full_emb - emb)
        
        # 床铺距离（特殊逻辑）
        current_bunk = current_user_dict.get('bunk_preference', '')
        other_bunk = user_details[i].get('bunk_preference', '')
        
        # 床铺匹配特殊逻辑
        bunk_dist = 2  # 默认距离
        if current_bunk == other_bunk:
            bunk_dist = 1  # 相同偏好
        elif (current_bunk == 'upper' and (other_bunk == 'lower' or other_bunk == '无')) or \
             (other_bunk == 'upper' and (current_bunk == 'lower' or current_bunk == '无')):
            bunk_dist = 0  # 上铺优先匹配下铺或无偏好
        
        # 睡眠重合度得分
        current_sleep = current_user_dict.get('sleep_time', '')
        current_wake = current_user_dict.get('wake_time', '')
        other_sleep = user_details[i].get('sleep_time', '')
        other_wake = user_details[i].get('wake_time', '')
        sleep_score = sleep_overlap_score(current_sleep, current_wake, other_sleep, other_wake)
        
        # 重新设计匹配算法：睡眠重合度最重要
        # 基础距离权重降低，睡眠重合度权重最高
        base_weight = 1  # 降低基础特征权重
        sleep_bonus = sleep_score * sleep_weight  # 睡眠重合度奖励（权重最高）
        bunk_penalty = bunk_dist * bunk_weight   # 床铺不匹配惩罚
        
        # 总距离 = 基础距离 + 床铺惩罚 - 睡眠奖励
        total_dist = base_weight * dist + bunk_penalty - sleep_bonus
        dists.append(total_dist)
    top3_idx = np.argsort(dists)[:3]
    return user_ids_valid[top3_idx].tolist() 