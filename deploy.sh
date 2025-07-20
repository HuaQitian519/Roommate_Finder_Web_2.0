#!/bin/bash

echo "🚀 开始部署室友匹配系统..."

# 停止并删除现有容器
echo "📦 停止现有容器..."
docker-compose down

# 清理可能损坏的容器和镜像
echo "🧹 清理损坏的容器和镜像..."
docker system prune -f

# 删除可能损坏的容器
echo "🗑️ 删除可能损坏的容器..."
docker container prune -f

# 删除可能损坏的镜像
echo "🗑️ 删除可能损坏的镜像..."
docker image prune -f

# 拉取最新镜像
echo "⬇️ 拉取最新镜像..."
docker-compose pull

# 重新构建镜像（如果需要）
echo "🔨 重新构建镜像..."
docker-compose build --no-cache

# 启动服务
echo "🚀 启动服务..."
docker-compose up -d

# 检查服务状态
echo "📊 检查服务状态..."
docker-compose ps

echo "✅ 部署完成！"
echo "🌐 访问地址: http://your-server-ip:8000" 