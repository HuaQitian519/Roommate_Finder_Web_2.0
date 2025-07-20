# 使用官方 Python 镜像作为基础镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 安装curl用于健康检查
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

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

# 创建启动脚本
RUN echo '#!/bin/bash\npython database.py\nexec gunicorn -w 4 -b 0.0.0.0:8000 app:app' > /app/start.sh && chmod +x /app/start.sh

# 使用启动脚本
CMD ["/app/start.sh"]