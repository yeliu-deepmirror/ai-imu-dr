FROM aliyunregistry.deepmirror.com.cn/dm/uranus-dev:presubmit-39f860e9930ee0dddb6e3f2c1385374c97cdcfff

RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
    matplotlib \
    termcolor \
    scipy \
    navpy
