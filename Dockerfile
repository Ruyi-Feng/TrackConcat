FROM yandy0725/pytorch:base

COPY requirements.txt /tmp/requirements.txt

RUN pip install --disable-pip-version-check --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/ -r /tmp/requirements.txt