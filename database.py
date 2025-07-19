from app import app, db
from models import User
import sqlite3
from werkzeug.security import generate_password_hash
from datetime import datetime

with app.app_context():
    db.create_all()

def migrate_add_mbti():
    conn = sqlite3.connect('database/roommate.db')
    cursor = conn.cursor()
    try:
        cursor.execute("ALTER TABLE user ADD COLUMN mbti TEXT")
    except Exception as e:
        print('MBTI字段可能已存在:', e)
    try:
        cursor.execute("ALTER TABLE user ADD COLUMN last_login DATETIME")
    except Exception as e:
        print('last_login字段可能已存在:', e)
    try:
        cursor.execute("ALTER TABLE user ADD COLUMN auto_set_roommate BOOLEAN DEFAULT 0")
    except Exception as e:
        print('auto_set_roommate字段可能已存在:', e)
    conn.commit()
    conn.close()

def init_admin():
    """初始化管理员账号"""
    with app.app_context():
        # 检查是否已存在管理员账号
        admin = User.query.filter_by(username='admin').first()
        if not admin:
            admin = User(
                username='admin',
                real_name='管理员',
                gender='男',
                major='智能科学与技术',
                sleep_time='23:00',
                wake_time='07:00',
                ac_temp=24,
                cleanliness=5,
                bunk_preference='上铺',
                study_habits='图书馆学习',
                roommate_preferences='希望室友安静、爱干净',
                weekend_stay='经常',
                group_life='是',
                hobbies='编程,阅读',
                dietary_restrictions='无',
                wechat='admin_wechat',
                ip_address='127.0.0.1',
                mbti='INTJ',
                is_admin=True,
                last_login=datetime.now()
            )
            admin.set_password('admin123')
            db.session.add(admin)
            db.session.commit()
            print('✅ 管理员账号已创建: admin / admin123')
        else:
            print('ℹ️ 管理员账号已存在')

if __name__ == '__main__':
    migrate_add_mbti()
    init_admin()
