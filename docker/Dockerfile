FROM ghcr.io/greenroom-robotics/ros_builder:jazzy-latest-cuda

COPY rtdetrv2_pytorch/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt --user 

RUN sudo apt-get update && sudo apt-get install -y python3-opencv

# Misc dependencies
RUN pip install debugpy

CMD ["tail", "-f", "/dev/null"]