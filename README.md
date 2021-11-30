# Bayesian Uncertainty for Quality Assessment of Deep Learning Contours
This repository contains Tensorflow2.4 code for the paper(s)
 - Comparing Bayesian Models for Organ Contouring in Headand Neck Radiotherapy


## Installation
1. Install [Anaconda](https://docs.anaconda.com/anaconda/install/) with python3.7
2. Install [git](https://git-scm.com/downloads)
3. Open a terminal and follow the commands
    - Clone this repository
        - `git clone git@github.com:prerakmody/hansegmentation-uncertainty-qa.git`
    - Create conda env
        - (Specifically For Windows): `conda init powershell` (and restart the terminal)
        - (For all plaforms)
        ```
        cd hansegmentation-uncertainty-qa
        conda deactivate
        conda create --name hansegmentation-uncertainty-qa python=3.8
        conda activate hansegmentation-uncertainty-qa
        conda develop .  # check for conda.pth file in $ANACONDA_HOME/envs/hansegmentation-uncertainty-qa/lib/python3.8/site-packages
        ```
    - Install packages
        - Tensorflow (check [here]((https://www.tensorflow.org/install/source#tested_build_configurations)) for CUDA/cuDNN requirements)
            - (stick to the exact commands) 
            - For tensorflow2.4
            ```
            conda install -c nvidia cudnn=8.0.0=cuda11.0_0
            pip install tensorflow==2.4
            ```
            - Check tensorflow installation
            ```
            python -c "import tensorflow as tf;print('\n\n\n====================== \n GPU Devices: ',tf.config.list_physical_devices('GPU'), '\n======================')"
            python -c "import tensorflow as tf;print('\n\n\n====================== \n', tf.reduce_sum(tf.random.normal([1000, 1000])), '\n======================' )"
            ```
                - [unix] upon running either of the above commands, you will see tensorflow searching for library files like libcudart.so, libcublas.so, libcublasLt.so, libcufft.so, libcurand.so, libcusolver.so, libcusparse.so, libcudnn.so in the location `$ANACONDA_HOME/envs/hansegmentation-uncertainty-qa/lib/`
                - [windows] upon running either of the above commands, you will see tensorflow searching for library files like cudart64_110.dll ... and so on in the location `$ANACONDA_HOME\envs\hansegmentation-uncertainty-qa\Library\bin`

            - Other tensorflow pacakges
            ```
            pip install tensorflow-probability==0.12.1 tensorflow-addons==0.12.1
            ```
        - Other packages
            ```
            pip install scipy seaborn tqdm psutil humanize pynrrd pydicom SimpleITK itk scikit-image
            pip install psutil humanize pynvml
            ```

# Notes
 - All the `src/train{}.py` files are the ones used to train the models as shown in the `demo/` folder