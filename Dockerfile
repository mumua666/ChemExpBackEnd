# 使用Python作为基础镜像
# FROM python:3.10
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app
# 复制当前目录下的所有文件到工作目录
COPY . /app
# 安装Flask和其他依赖
# 清华源: https://pypi.tuna.tsinghua.edu.cn/simple
# 阿里源: https://mirrors.aliyun.com/pypi/simple
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade pip
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r /app/requirements.txt
# 带缓存安装依赖文件
# RUN pip install -i  https://mirrors.aliyun.com/pypi/simple --cache-dir=/tmp/pip-cache -r /app/requirements.txt

# RUN apt update
# RUN apt install -y libgl1-mesa-glx

# 使用镜像下载安装依赖
RUN sed -i 's|http://deb.debian.org/debian|http://mirrors.aliyun.com/debian|g' /etc/apt/sources.list && \
    apt update && \
    apt install -y --no-install-recommends libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*


# 暴露端口
EXPOSE 3729
# 运行应用
CMD ["python", "app.py"]


# 构建镜像
# docker build -t flask-docker .
# 运行容器
# docker run -d -p 3729:3730 flask-docker
