#!/bin/bash

# 构建x86镜像
echo "正在构建x86镜像..."
docker build --platform linux/amd64 -t huaqitian/roommate_matcher:latest .

# 登录Docker Hub
echo "请登录Docker Hub（如已登录可忽略）"
docker login

# 推送镜像
echo "正在推送镜像到Docker Hub..."
docker push huaqitian/roommate_matcher:latest

echo "请将本项目文件夹（含docker-compose.yml）上传到服务器后，执行："
echo "docker pull huaqitian/roommate_matcher:latest"
echo "docker-compose up -d"
echo "访问 http://服务器IP:8000 即可" 