echo "******** Installing Conda packages ********"
conda install -y \
    @EXPLICIT \
    https://repo.anaconda.com/pkgs/main/linux-64/_libgcc_mutex-0.1-main.conda \
    https://repo.anaconda.com/pkgs/main/linux-64/blas-1.0-mkl.conda \
    https://repo.anaconda.com/pkgs/main/linux-64/ca-certificates-2023.12.12-h06a4308_0.conda \
    https://repo.anaconda.com/pkgs/main/linux-64/intel-openmp-2021.4.0-h06a4308_3561.conda \
    https://repo.anaconda.com/pkgs/main/linux-64/ld_impl_linux-64-2.38-h1181459_1.conda \
    https://repo.anaconda.com/pkgs/main/linux-64/libgfortran5-11.2.0-h1234567_1.conda \
    https://repo.anaconda.com/pkgs/main/linux-64/libstdcxx-ng-11.2.0-h1234567_1.conda \
    https://conda.anaconda.org/pytorch/noarch/pytorch-mutex-1.0-cuda.tar.bz2 \
    https://repo.anaconda.com/pkgs/main/linux-64/libgfortran-ng-11.2.0-h00389a5_1.conda \
    https://repo.anaconda.com/pkgs/main/linux-64/libgomp-11.2.0-h1234567_1.conda \
    https://repo.anaconda.com/pkgs/main/linux-64/mkl-2021.4.0-h06a4308_640.conda \
    https://repo.anaconda.com/pkgs/main/linux-64/_openmp_mutex-5.1-1_gnu.conda \
    https://repo.anaconda.com/pkgs/main/linux-64/libgcc-ng-11.2.0-h1234567_1.conda \
    https://repo.anaconda.com/pkgs/main/linux-64/bzip2-1.0.8-h7b6447c_0.conda \
    https://repo.anaconda.com/pkgs/main/linux-64/cudatoolkit-11.3.1-h2bc3f7f_2.conda \
    https://repo.anaconda.com/pkgs/main/linux-64/giflib-5.2.1-h5eee18b_3.conda \
    https://repo.anaconda.com/pkgs/main/linux-64/gmp-6.2.1-h295c915_3.conda \
    https://repo.anaconda.com/pkgs/main/linux-64/jpeg-9e-h5eee18b_1.conda \
    https://repo.anaconda.com/pkgs/main/linux-64/lame-3.100-h7b6447c_0.conda \
    https://repo.anaconda.com/pkgs/main/linux-64/lerc-3.0-h295c915_0.conda \
    https://repo.anaconda.com/pkgs/main/linux-64/libdeflate-1.17-h5eee18b_1.conda \
    https://repo.anaconda.com/pkgs/main/linux-64/libffi-3.3-he6710b0_2.conda \
    https://repo.anaconda.com/pkgs/main/linux-64/libiconv-1.16-h7f8727e_2.conda \
    https://repo.anaconda.com/pkgs/main/linux-64/libtasn1-4.19.0-h5eee18b_0.conda \
    https://repo.anaconda.com/pkgs/main/linux-64/libunistring-0.9.10-h27cfd23_0.conda \
    https://repo.anaconda.com/pkgs/main/linux-64/libwebp-base-1.3.2-h5eee18b_0.conda \
    https://repo.anaconda.com/pkgs/main/linux-64/lz4-c-1.9.4-h6a678d5_0.conda \
    https://repo.anaconda.com/pkgs/main/linux-64/ncurses-6.4-h6a678d5_0.conda \
    https://repo.anaconda.com/pkgs/main/linux-64/openh264-2.1.1-h4ff587b_0.conda \
    https://repo.anaconda.com/pkgs/main/linux-64/openssl-1.1.1w-h7f8727e_0.conda \
    https://repo.anaconda.com/pkgs/main/linux-64/xz-5.4.5-h5eee18b_0.conda \
    https://repo.anaconda.com/pkgs/main/linux-64/zlib-1.2.13-h5eee18b_0.conda \
    https://repo.anaconda.com/pkgs/main/linux-64/libidn2-2.3.4-h5eee18b_0.conda \
    https://repo.anaconda.com/pkgs/main/linux-64/libpng-1.6.39-h5eee18b_0.conda \
    https://repo.anaconda.com/pkgs/main/linux-64/nettle-3.7.3-hbbd107a_1.conda \
    https://repo.anaconda.com/pkgs/main/linux-64/readline-8.2-h5eee18b_0.conda \
    https://repo.anaconda.com/pkgs/main/linux-64/tk-8.6.12-h1ccaba5_0.conda \
    https://repo.anaconda.com/pkgs/main/linux-64/zstd-1.5.5-hc292b87_0.conda \
    https://repo.anaconda.com/pkgs/main/linux-64/freetype-2.12.1-h4a9f257_0.conda \
    https://repo.anaconda.com/pkgs/main/linux-64/gnutls-3.6.15-he1e5248_0.conda \
    https://repo.anaconda.com/pkgs/main/linux-64/libtiff-4.5.1-h6a678d5_0.conda \
    https://repo.anaconda.com/pkgs/main/linux-64/sqlite-3.41.2-h5eee18b_0.conda \
    https://conda.anaconda.org/pytorch/linux-64/ffmpeg-4.3-hf484d3e_0.tar.bz2 \
    https://repo.anaconda.com/pkgs/main/linux-64/lcms2-2.12-h3be6417_0.conda \
    https://repo.anaconda.com/pkgs/main/linux-64/libwebp-1.3.2-h11a3e52_0.conda \
    https://repo.anaconda.com/pkgs/main/linux-64/openjpeg-2.4.0-h3ad879b_0.conda \
    https://repo.anaconda.com/pkgs/main/linux-64/python-3.8.5-h7579374_1.conda \
    https://repo.anaconda.com/pkgs/main/linux-64/brotli-python-1.0.9-py38h6a678d5_7.conda \
    https://repo.anaconda.com/pkgs/main/linux-64/certifi-2023.11.17-py38h06a4308_0.conda \
    https://repo.anaconda.com/pkgs/main/noarch/charset-normalizer-2.0.4-pyhd3eb1b0_0.conda \
    https://repo.anaconda.com/pkgs/main/linux-64/idna-3.4-py38h06a4308_0.conda \
    https://repo.anaconda.com/pkgs/main/linux-64/packaging-23.1-py38h06a4308_0.conda \
    https://repo.anaconda.com/pkgs/main/linux-64/pillow-10.0.1-py38ha6cbd5a_0.conda \
    https://repo.anaconda.com/pkgs/main/linux-64/platformdirs-3.10.0-py38h06a4308_0.conda \
    https://repo.anaconda.com/pkgs/main/noarch/pycparser-2.21-pyhd3eb1b0_0.conda \
    https://repo.anaconda.com/pkgs/main/linux-64/pysocks-1.7.1-py38h06a4308_0.conda \
    https://conda.anaconda.org/pytorch/linux-64/pytorch-1.12.1-py3.8_cuda11.3_cudnn8.3.2_0.tar.bz2 \
    https://repo.anaconda.com/pkgs/main/linux-64/setuptools-68.2.2-py38h06a4308_0.conda \
    https://repo.anaconda.com/pkgs/main/noarch/six-1.16.0-pyhd3eb1b0_1.conda \
    https://repo.anaconda.com/pkgs/main/linux-64/tqdm-4.65.0-py38hb070fc8_0.conda \
    https://repo.anaconda.com/pkgs/main/linux-64/wheel-0.41.2-py38h06a4308_0.conda \
    https://repo.anaconda.com/pkgs/main/linux-64/cffi-1.15.1-py38h74dc2b5_0.conda \
    https://repo.anaconda.com/pkgs/main/linux-64/mkl-service-2.4.0-py38h7f8727e_0.conda \
    https://repo.anaconda.com/pkgs/main/linux-64/pip-20.3.3-py38h06a4308_0.conda \
    https://repo.anaconda.com/pkgs/main/linux-64/cryptography-41.0.3-py38h130f0dd_0.conda \
    https://repo.anaconda.com/pkgs/main/linux-64/numpy-base-1.23.1-py38ha15fc14_0.conda \
    https://repo.anaconda.com/pkgs/main/linux-64/pyopenssl-23.2.0-py38h06a4308_0.conda \
    https://repo.anaconda.com/pkgs/main/linux-64/urllib3-1.26.18-py38h06a4308_0.conda \
    https://repo.anaconda.com/pkgs/main/linux-64/requests-2.31.0-py38h06a4308_0.conda \
    https://repo.anaconda.com/pkgs/main/linux-64/pooch-1.7.0-py38h06a4308_0.conda \
    https://repo.anaconda.com/pkgs/main/linux-64/mkl_fft-1.3.1-py38hd3c417c_0.conda \
    https://repo.anaconda.com/pkgs/main/linux-64/mkl_random-1.2.2-py38h51133e4_0.conda \
    https://repo.anaconda.com/pkgs/main/linux-64/numpy-1.23.1-py38h6c91a56_0.conda \
    https://repo.anaconda.com/pkgs/main/linux-64/scipy-1.10.0-py38h14f4228_1.conda \
    https://conda.anaconda.org/pytorch/linux-64/torchvision-0.13.1-py38_cu113.tar.bz2 \

echo ""
echo "******** Installing Pip packages ********"
pip install absl-py==1.2.0
pip install actionlib==1.14.0
pip install aiohttp==3.9.5
pip install aiosignal==1.3.1
pip install altair==5.4.1
pip install angles==1.9.13
pip install annotated-types==0.6.0
pip install antlr4-python3-runtime==4.9.3
pip install argon2-cffi==21.3.0
pip install argon2-cffi-bindings==21.2.0
pip install asttokens==2.0.8
pip install astunparse==1.6.3
pip install async-timeout==4.0.3
pip install attrs==24.3.0
pip install backcall==0.2.0
pip install beautifulsoup4==4.11.1
pip install bleach==5.0.1
pip install blend-modes==2.1.0
pip install blinker==1.8.2
pip install blis==0.7.11
pip install bondpy==1.8.6
pip install braceexpand==0.1.7
pip install Brotli @ file:///tmp/abs_ecyw11_7ze/croots/recipe/brotli-split_1659616059936/work
pip install camera-calibration==1.17.0
pip install camera-calibration-parsers==1.12.0
pip install catalogue==2.0.10
pip install catkin==0.8.10
pip install certifi==2024.8.30
pip install cffi==1.15.1
pip install cfgv==3.4.0
pip install charset-normalizer @ file:///tmp/build/80754af9/charset-normalizer_1630003229654/work
pip install click==8.1.7
pip install clip @ git+https://github.com/openai/CLIP.git@a1d071733d7111c9c014f024669f959182114e33
pip install cloudpathlib==0.16.0
pip install colorama==0.4.6
pip install comm==0.2.1
pip install confection==0.1.4
pip install contexttimer==0.3.3
pip install contourpy==1.1.1
pip install controller-manager==0.20.0
pip install controller-manager-msgs==0.20.0
pip install cryptography @ file:///croot/cryptography_1694444244250/work
pip install cv-bridge==1.16.2
pip install cycler==0.12.1
pip install cymem==2.0.8
pip install datasets==2.19.1
pip install debugpy==1.6.3
pip install decorator==5.1.1
pip install decord==0.6.0
pip install defusedxml==0.7.1
pip install diagnostic-analysis==1.11.0
pip install diagnostic-common-diagnostics==1.11.0
pip install diagnostic-updater==1.11.0
pip install dill==0.3.8
pip install distlib==0.3.8
pip install dlib==19.24.0
pip install docker-pycreds==0.4.0
pip install dynamic-reconfigure==1.7.3
pip install easydict==1.11
pip install einops==0.8.0
pip install en-core-web-md @ https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.7.1/en_core_web_md-3.7.1-py3-none-any.whl
pip install en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl
pip install entrypoints==0.4
pip install executing==1.1.1
pip install face-recognition==1.3.0
pip install face-recognition-models==0.3.0
pip install fairscale==0.4.4
pip install fastjsonschema==2.16.2
pip install filelock==3.13.1
pip install fonttools==4.55.3
pip install frozenlist==1.4.1
pip install fsspec==2023.12.2
pip install ftfy==6.1.3
pip install gast==0.4.0
pip install gazebo-plugins==2.9.3
pip install gazebo-ros==2.9.3
pip install gdown==5.2.0
pip install gencpp==0.7.0
pip install geneus==3.0.0
pip install genlisp==0.4.18
pip install genmsg==0.6.0
pip install gennodejs==2.0.2
pip install genpy==0.6.15
pip install gitdb==4.0.11
pip install GitPython==3.1.43
pip install google-auth==2.35.0
pip install google-auth-oauthlib==1.0.0
pip install google-pasta==0.2.0
pip install grpcio==1.49.1
pip install h5py==3.7.0
pip install huggingface-hub==0.23.0
pip install identify==2.6.1
pip install idna @ file:///croot/idna_1666125576474/work
pip install image-geometry==1.16.2
pip install imageio==2.35.1
pip install imagenetv2-pytorch @ git+https://github.com/modestyachts/ImageNetV2_pytorch@14d4456c39fe7f02a665544dd9fc37c1a5f8b635
pip install importlib-metadata==8.5.0
pip install importlib-resources==6.4.5
pip install imutils==0.5.4
pip install interactive-markers==1.12.0
pip install iopath==0.1.10
pip install ipykernel==6.16.0
pip install ipython==8.5.0
pip install ipython-genutils==0.2.0
pip install ipywidgets==8.1.1
pip install jedi==0.18.1
pip install Jinja2==3.1.2
pip install joblib==1.4.2
pip install joint-state-publisher==1.15.1
pip install joint-state-publisher-gui==1.15.1
pip install jpeg4py==0.1.4
pip install jsonargparse==4.21.0
pip install jsonpatch==1.33
pip install jsonpointer==2.4
pip install jsonschema==4.23.0
pip install jsonschema-specifications==2023.12.1
pip install jupyter==1.0.0
pip install jupyter-client==7.3.5
pip install jupyter-console==6.6.3
pip install jupyter-core==5.7.1
pip install jupyterlab-pygments==0.2.2
pip install jupyterlab-widgets==3.0.9
pip install kaggle==1.6.17
pip install Keras-Preprocessing==1.1.2
pip install kiwisolver==1.4.5
pip install langcodes==3.4.0
pip install language-data==1.2.0
pip install laser-geometry==1.6.7
pip install lazy-loader==0.4
pip install libclang==14.0.6
pip install lightning-utilities==0.10.0
pip install littleutils==0.2.4
pip install llvmlite==0.41.1
pip install lmdb==1.4.1
pip install loguru==0.7.3
pip install marisa-trie==1.1.1
pip install Markdown==3.4.1
pip install markdown-it-py==3.0.0
pip install MarkupSafe==2.1.1
pip install matplotlib==3.7.5
pip install matplotlib-inline==0.1.6
pip install mdurl==0.1.2
pip install message-filters==1.17.0
pip install mistune==2.0.4
pip install mkl-fft==1.3.1
pip install mkl-random @ file:///tmp/build/80754af9/mkl_random_1626186064646/work
pip install mkl-service==2.4.0
pip install multidict==6.0.5
pip install multiprocess==0.70.16
pip install murmurhash==1.0.10
pip install narwhals==1.8.4
pip install nbclient==0.7.0
pip install nbconvert==7.2.1
pip install nbformat==5.10.4
pip install nest-asyncio==1.5.6
pip install networkx==3.1
pip install nltk==3.8.1
pip install nodeenv==1.9.1
pip install notebook==6.4.12
pip install numba==0.58.1
pip install numpy @ file:///tmp/abs_653_j00fmm/croots/recipe/numpy_and_numpy_base_1659432701727/work
pip install nvidia-cuda-nvrtc-cu11==11.7.99
pip install nvidia-pyindex==1.0.9
pip install oauthlib==3.2.2
pip install ogb==1.3.6
pip install omegaconf==2.3.0
pip install opencv-python==4.10.0.84
pip install opencv-python-headless==4.5.5.64
pip install opendatasets==0.1.22
pip install opt-einsum==3.3.0
pip install outdated==0.2.2
pip install packaging @ file:///croot/packaging_1693575174725/work
pip install pandas==2.0.3
pip install pandocfilters==1.5.0
pip install parso==0.8.3
pip install pexpect==4.8.0
pip install pickleshare==0.7.5
pip install Pillow @ file:///croot/pillow_1696580024257/work
pip install pkgutil-resolve-name==1.3.10
pip install platformdirs @ file:///croot/platformdirs_1692205439124/work
pip install plotly==5.24.1
pip install pooch @ file:///croot/pooch_1695850093751/work
pip install portalocker==2.10.1
pip install pre-commit==3.5.0
pip install preshed==3.0.9
pip install prometheus-client==0.14.1
pip install prompt-toolkit==3.0.31
pip install protobuf==5.28.2
pip install psutil==5.9.2
pip install ptyprocess==0.7.0
pip install pure-eval==0.2.2
pip install pyarrow==16.1.0
pip install pyarrow-hotfix==0.6
pip install pyasn1==0.4.8
pip install pyasn1-modules==0.2.8
pip install pycocoevalcap==1.2
pip install pycocotools==2.0.7
pip install pycparser==2.21
pip install pydantic==2.7.1
pip install pydantic-core==2.18.2
pip install pydeck==0.9.1
pip install Pygments==2.13.0
pip install pyOpenSSL @ file:///croot/pyopenssl_1690223430423/work
pip install pyparsing==3.1.4
pip install pyrsistent==0.18.1
pip install PySocks @ file:///tmp/build/80754af9/pysocks_1605305779399/work
pip install python-dateutil==2.9.0.post0
pip install python-magic==0.4.27
pip install python-qt-binding==0.4.4
pip install python-slugify==8.0.4
pip install pytz==2022.7.1
pip install PyWavelets==1.4.1
pip install PyYAML==6.0.1
pip install pyzmq==24.0.1
pip install qt-dotgraph==0.4.2
pip install qt-gui==0.4.2
pip install qt-gui-cpp==0.4.2
pip install qt-gui-py-common==0.4.2
pip install qtconsole==5.5.1
pip install QtPy==2.4.1
pip install referencing==0.35.1
pip install regex==2023.12.25
pip install requests @ file:///croot/requests_1690400202158/work
pip install requests-oauthlib==1.3.1
pip install resource-retriever==1.12.8
pip install rich==13.8.1
pip install rosbag==1.17.0
pip install rosboost-cfg==1.15.8
pip install rosclean==1.15.8
pip install roscreate==1.15.8
pip install rosgraph==1.17.0
pip install roslaunch==1.17.0
pip install roslib==1.15.8
pip install roslint==0.12.0
pip install roslz4==1.17.0
pip install rosmake==1.15.8
pip install rosmaster==1.17.0
pip install rosmsg==1.17.0
pip install rosnode==1.17.0
pip install rosparam==1.17.0
pip install rospy==1.17.0
pip install rosservice==1.17.0
pip install rostest==1.17.0
pip install rostopic==1.17.0
pip install rosunit==1.15.8
pip install roswtf==1.17.0
pip install rpds-py==0.20.1
pip install rqt-action==0.4.9
pip install rqt-bag==0.5.1
pip install rqt-bag-plugins==0.5.1
pip install rqt-console==0.4.12
pip install rqt-dep==0.4.12
pip install rqt-graph==0.4.14
pip install rqt-gui==0.5.3
pip install rqt-gui-py==0.5.3
pip install rqt-image-view==0.4.17
pip install rqt-launch==0.4.9
pip install rqt-logger-level==0.4.12
pip install rqt-moveit==0.5.11
pip install rqt-msg==0.4.10
pip install rqt-nav-view==0.5.7
pip install rqt-plot==0.4.13
pip install rqt-pose-view==0.5.11
pip install rqt-publisher==0.4.10
pip install rqt-py-common==0.5.3
pip install rqt-py-console==0.4.10
pip install rqt-reconfigure==0.5.5
pip install rqt-robot-dashboard==0.5.8
pip install rqt-robot-monitor==0.5.15
pip install rqt-robot-steering==0.5.12
pip install rqt-runtime-monitor==0.5.10
pip install rqt-rviz==0.7.0
pip install rqt-service-caller==0.4.10
pip install rqt-shell==0.4.11
pip install rqt-srv==0.4.9
pip install rqt-tf-tree==0.6.4
pip install rqt-top==0.4.10
pip install rqt-topic==0.4.13
pip install rqt-web==0.4.10
pip install rsa==4.9
pip install rviz==1.14.25
pip install safetensors==0.4.1
pip install salesforce-lavis==1.0.2
pip install scikit-image==0.21.0
pip install scikit-learn==1.3.2
pip install scipy==1.10.0
pip install seaborn==0.12.2
pip install Send2Trash==1.8.0
pip install sensor-msgs==1.13.1
pip install sentencepiece==0.2.0
pip install sentry-sdk==2.19.2
pip install setproctitle==1.3.3
pip install shapely==2.0.7
pip install six @ file:///tmp/build/80754af9/six_1644875935023/work
pip install smach==2.5.3
pip install smach-ros==2.5.3
pip install smart-open==6.4.0
pip install smclib==1.8.6
pip install smmap==5.0.1
pip install soupsieve==2.3.2.post1
pip install spacy==3.7.4
pip install spacy-legacy==3.0.12
pip install spacy-loggers==1.0.5
pip install srsly==2.4.8
pip install stack-data==0.5.1
pip install streamlit==1.38.0
pip install tabulate==0.9.0
pip install tenacity==8.5.0
pip install tensorboard==2.14.0
pip install tensorboard-data-server==0.7.2
pip install tensorboard-plugin-wit==1.8.1
pip install tensorboardX==2.5.1
pip install tensorflow-io-gcs-filesystem==0.27.0
pip install termcolor==2.0.1
pip install terminado==0.16.0
pip install text-unidecode==1.3
pip install tf==1.13.2
pip install tf-conversions==1.13.2
pip install tf2-geometry-msgs==0.7.7
pip install tf2-kdl==0.7.7
pip install tf2-py==0.7.7
pip install tf2-ros==0.7.7
pip install thinc==8.2.3
pip install thop==0.1.1.post2209072238
pip install threadpoolctl==3.5.0
pip install tifffile==2023.7.10
pip install tikzplotlib==0.10.1
pip install timm==0.4.12
pip install tinycss2==1.1.1
pip install tokenizers==0.13.3
pip install toml==0.10.2
pip install topic-tools==1.17.0
pip install torch==1.12.1
pip install torch-tb-profiler==0.4.3
pip install torchmetrics==0.6.0
pip install torchvision==0.13.1
pip install tornado==6.2
pip install tqdm @ file:///croot/tqdm_1679561862951/work
pip install traitlets==5.4.0
pip install transformers==4.26.1
pip install typer==0.9.4
pip install typing-extensions==4.12.2
pip install tzdata==2024.2
pip install ultralytics==8.0.20
pip install urllib3 @ file:///croot/urllib3_1698257533958/work
pip install virtualenv==20.26.6
pip install visdom==0.2.4
pip install wandb==0.17.1
pip install wasabi==1.1.2
pip install watchdog==4.0.2
pip install wcwidth==0.2.13
pip install weasel==0.3.4
pip install webcolors==24.6.0
pip install webdataset==0.2.100
pip install webencodings==0.5.1
pip install websocket-client==1.7.0
pip install werkzeug==3.0.4
pip install widgetsnbextension==4.0.9
pip install wilds==2.0.0
pip install wrapt==1.14.1
pip install xacro==1.14.19
pip install xxhash==3.4.1
pip install yacs==0.1.8
pip install yarl==1.9.4
pip install zipp==3.20.2

echo ""
echo "******** Installation complete! ********"