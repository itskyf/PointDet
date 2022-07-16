FROM mambaorg/micromamba:latest as builder

# Split PyTorch to use with runtime
RUN micromamba install -y -n base -c pytorch -c conda-forge \
	cudatoolkit=11.6 pytorch python=3.9 &&\
	micromamba clean -y --all

RUN micromamba install -y -n base -c nvidia/label/cuda-11.6.2 -c conda-forge \
	cuda-toolkit gxx=11 git ninja pip setuptools &&\
	micromamba clean -y --all

ARG CUDA_ARCH
ARG MAMBA_DOCKERFILE_ACTIVATE=1
RUN cd ~ && git clone --depth 1 https://github.com/itskyf/PointDet &&\
	TORCH_CUDA_ARCH_LIST="$CUDA_ARCH" FORCE_CUDA=1 pip install \
	--no-build-isolation --no-cache-dir --no-deps --upgrade --editable ./PointDet &&\
	rm -rf ./PointDet/build

FROM mambaorg/micromamba:latest

# Split PyTorch to use with runtime
RUN micromamba install -y -n base -c pytorch -c conda-forge \
	cudatoolkit=11.6 pytorch python=3.9 &&\
	micromamba clean -y --all

RUN micromamba install -y --name base --channel conda-forge \
	hydra-core numba scipy tensorboard tqdm &&\
	micromamba install -y --name base --channel conda-forge gdown git rclone tmux vim &&\
	micromamba shell init --shell=bash --prefix=$MAMBA_ROOT_PREFIX &&\
	micromamba clean -y --all

USER root
RUN --mount=target=/tmp/context,source=. \
	install -m755 /tmp/context/start.sh ~ &&\
	apt-get update &&\
	apt-get install -y --no-install-recommends openssh-server &&\
	echo 'set -g mouse on' >> ~/.tmux.conf &&\
	echo 'setw -g mode-keys vi' >> ~/.tmux.conf &&\
	echo 'bind Escape copy-mode' >> ~/.tmux.conf &&\
	rm -rf /var/lib/apt/lists/*

ARG MAMBA_DOCKERFILE_ACTIVATE=1
RUN --mount=target=/tmp/builder,source=/home/mambauser,from=builder \
	cp -r /tmp/builder/PointDet ~ &&\
	echo "$HOME/PointDet/src" > $MAMBA_ROOT_PREFIX/lib/python3.9/site-packages/easy-install.pth &&\
	printf "$HOME/PointDet/src\n../" > $MAMBA_ROOT_PREFIX/lib/python3.9/site-packages/pointdet.egg-link

WORKDIR /root/PointDet/tools
CMD ["~/start.sh"]
