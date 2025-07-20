# 室友匹配系统

一个基于 Flask 和机器学习的智能室友匹配系统，帮助大学生找到合适的室友。

## 🌟 功能特性

### 核心功能
- **智能匹配算法**：基于用户生活习惯、作息时间、专业等特征进行精准匹配
- **BERT 语义分析**：使用 BERT 模型分析用户爱好，提升匹配精度
- **实时匹配**：支持实时匹配和随机推荐
- **用户管理**：完整的用户注册、登录、信息管理功能

### 用户体验
- **响应式设计**：支持手机端和桌面端访问
- **表单数据保留**：注册失败时保留已填写信息
- **用户协议**：完整的隐私保护和使用协议
- **管理员面板**：用户审核、封禁、删除等管理功能

### 技术特性
- **Docker 部署**：一键部署，支持 Docker Hub 拉取
- **数据库迁移**：自动数据库初始化和字段迁移
- **安全防护**：IP 限制、登录尝试限制、数据加密
- **自动清理**：3天未登录用户自动标记为已找到室友

## 🚀 快速开始

### 使用 Docker 部署（推荐）

1. **拉取镜像**
```bash
docker pull huaqitian/roommate_matcher:latest
```

2. **创建 docker-compose.yml**
```yaml
version: '3.8'
services:
  roommate_matcher:
    image: huaqitian/roommate_matcher:latest
    ports:
      - "8000:8000"
    volumes:
      - ./database:/app/database
    restart: unless-stopped
```

3. **启动服务**
```bash
docker-compose up -d
```

4. **访问系统**
```
http://localhost:8000
```

### 本地开发部署

1. **克隆项目**
```bash
git clone https://github.com/your-username/roommate_matcher.git
cd roommate_matcher
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **初始化数据库**
```bash
python database.py
```

4. **启动服务**
```bash
python app.py
```

## 📋 系统要求

- Python 3.9+
- SQLite 数据库
- 2GB+ 内存
- 支持 Docker（可选）

## 🔧 配置说明

### 环境变量
- `FLASK_ENV`: 运行环境（production/development）
- `FLASK_APP`: 应用入口文件

### 数据库配置
- 默认使用 SQLite 数据库
- 数据库文件位置：`database/roommate.db`
- 支持自动迁移和初始化

## 👥 用户指南

### 注册流程
1. 访问首页，点击"立即注册"
2. 填写个人信息（姓名、专业、作息时间等）
3. 阅读并同意用户协议
4. 提交注册，等待管理员审核

### 匹配功能
1. 登录后点击"智能匹配"
2. 可选择"只显示本专业同学"
3. 查看匹配结果，复制微信号联系

### 管理员功能
- **默认管理员账号**：`admin` / `admin123`
- **用户审核**：审核新注册用户
- **用户管理**：封禁、删除、编辑用户信息

## 🛠️ 技术架构

### 后端技术栈
- **Flask**: Web 框架
- **SQLAlchemy**: ORM 数据库操作
- **Flask-Login**: 用户认证
- **scikit-learn**: 机器学习算法
- **transformers**: BERT 模型
- **torch**: 深度学习框架

### 前端技术栈
- **Bootstrap 5**: UI 框架
- **Jinja2**: 模板引擎
- **JavaScript**: 交互功能

### 部署技术
- **Docker**: 容器化部署
- **Gunicorn**: WSGI 服务器
- **Caddy**: 反向代理（可选）

## 📊 匹配算法

### 特征权重
- **作息时间**（睡觉/起床）: 权重 5（最高优先级）
- **卫生习惯**: 权重 3
- **学习习惯**: 权重 2
- **床铺偏好**: 权重 2（特殊匹配逻辑）
- **爱好**（BERT语义）: 权重 0.5
- **其他特征**: 权重 1

### 床铺偏好特殊逻辑
- 上铺优先匹配下铺或无偏好（距离为0）
- 相同偏好距离为1
- 其他情况距离为2

### BERT 语义分析
- 使用 `bert-base-chinese` 模型
- 分析用户爱好文本
- 计算语义相似度
- 提升匹配精度

## 🔒 安全特性

- **密码加密**: 使用 Werkzeug 安全哈希
- **IP 限制**: 单个 IP 只能注册一个账号
- **登录限制**: 24小时内最多10次登录尝试
- **数据保护**: 用户协议明确数据使用和销毁流程
- **权限控制**: 管理员权限验证

## 📝 开发说明

### 项目结构
```
roommate_matcher/
├── app.py                 # 主应用文件
├── models.py             # 数据模型
├── database.py           # 数据库初始化
├── embedding_utils.py    # 匹配算法
├── requirements.txt      # 依赖包
├── Dockerfile           # Docker 配置
├── docker-compose.yml   # Docker Compose 配置
├── templates/           # 前端模板
├── static/             # 静态文件
└── database/           # 数据库文件
```

### 添加新功能
1. 在 `models.py` 中定义数据模型
2. 在 `app.py` 中添加路由
3. 在 `templates/` 中创建模板
4. 更新 `database.py` 进行数据迁移

## 🤝 贡献指南

1. Fork 本项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📄 许可证

本项目采用 GPL V3.0 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情


## 🙏 致谢

感谢所有为这个项目做出贡献的开发者和用户！

---

**注意**: 本项目仅供学习和研究使用，请遵守相关法律法规和用户协议。 
