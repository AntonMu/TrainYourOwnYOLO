FROM python:3.7-slim
WORKDIR /workspace
ENV LC_ALL=C

RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  libgl1 \
  libglib2.0-0 \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# Copy entire TrainYourOwnYOLO code and install required packages
COPY . /workspace
RUN pip install -r requirements.txt
