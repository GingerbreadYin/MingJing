# CPU PyTorch 本地构建缓存

默认从官方 PyTorch CPU 索引下载。网络不稳定时，可以用 `RAGOPT_CPU_WHEELS` 指向一个仅放 CPU wheel 的本机目录。Dockerfile 会核对 Python 3.12 Linux x86-64 wheel 的 SHA-256，再将它安装到检索环境。
