# One image for every lab role. Only the integrity library and lab code are copied in:
# no .env, no data, no repo history. Each service gets its own few env vars from compose.
FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends iproute2 iptables tcpdump iputils-ping procps ethtool \
    && rm -rf /var/lib/apt/lists/*

COPY lab/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /app
ENV PYTHONPATH=/app PYTHONUNBUFFERED=1
COPY securetrace/__init__.py securetrace/__init__.py
COPY securetrace/integrity securetrace/integrity
COPY lab lab
