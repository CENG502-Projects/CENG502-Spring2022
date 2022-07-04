FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

RUN pip3 install \
  matplotlib \
  sklearn \
  h5py \
  higher \
  pyyaml==5.4.1

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub

# developer user definitions
ENV USERNAME=user502
ENV USER_UID=1000
ENV USER_GID=$USER_UID

RUN apt update && apt install -y git \
  && apt install bash-completion wget

RUN mkdir -p /etc/sudoers.d/

# Create the user
RUN groupadd --gid $USER_GID $USERNAME && useradd -m --uid $USER_UID --gid $USER_GID $USERNAME \
    # add sudo support
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && usermod -a -G dialout user502

#Switch to the non-root user
USER $USERNAME

CMD ["/bin/bash"]