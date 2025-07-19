# 使用官方 Python 镜像作为基础镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 复制项目的依赖文件
COPY requirements.txt .

# 安装项目的依赖（确保torch为CPU版）
RUN pip install --no-cache-dir -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html

# 复制项目文件到工作目录
COPY . .

# 确保数据库目录存在
RUN mkdir -p /app/database

# 设置环境变量
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# 暴露应用端口
EXPOSE 8000

# 启动时总是执行数据库初始化（包括管理员账号检查）
CMD python database.py && gunicorn -w 4 -b 0.0.0.0:8000 app:app