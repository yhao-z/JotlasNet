# JotlasNet (MRI 2025)
( If you find this code is helpful to you, could you please kindly give me a star <img src="https://slackmojis.com/emojis/13058-star_spin/download" width="30"/>)

The official TensorFlow implementation of **JotlasNet: Joint tensor low-rank and attention-based sparse unrolling network for accelerating dynamic MRI**, *Magnetic Resonance Imaging*, 2025. ([Journal](https://www.sciencedirect.com/science/article/pii/S0730725X25000190) | [arXiv](https://arxiv.org/abs/2502.11749))

## Abstract

Joint low-rank and sparse unrolling networks have shown superior performance in dynamic MRI reconstruction. However, existing works mainly utilized matrix low-rank priors, neglecting the tensor characteristics of dynamic MRI images, and only a global threshold is applied for the sparse constraint to the multi-channel data, limiting the flexibility of the network. Additionally, most of them have inherently complex network structure, with intricate interactions among variables. In this paper, we propose a novel deep unrolling network, JotlasNet, for dynamic MRI reconstruction by jointly utilizing tensor low-rank and attention-based sparse priors. Specifically, we utilize tensor low-rank prior to exploit the structural correlations in high-dimensional data. Convolutional neural networks are used to adaptively learn the low-rank and sparse transform domains. A novel attention-based soft thresholding operator is proposed to assign a unique learnable threshold to each channel of the data in the CNN-learned sparse domain. The network is unrolled from the elaborately designed composite splitting algorithm and thus features a simple yet efficient parallel structure.  Extensive experiments on two datasets (OCMR, CMRxRecon) demonstrate the superior performance of JotlasNet in dynamic MRI reconstruction. 

![](https://yhao-img-bed.obs.cn-north-4.myhuaweicloud.com/202502172008130.png)

## 1. Getting Started

### Environment Configuration

- we recommend to use docker

  ```shell
  # pull the docker images
  docker pull yhaoz/tf:2.9.0-bart
  # then you can create a container to run the code, see docker documents for more details
  ```

- if you don't have docker, you can still configure it via installing the requirements by yourself

  ```shell
  pip install -r requirements.txt # tensorflow is gpu version
  ```

Note that, we only run the code in NVIDIA GPU. In our implementation, the code can run normally in both Linux & Windows system.

### Dataset preparation

You can download the **single-coil** dataset via [my OneDrive](https://stuhiteducn-my.sharepoint.com/:f:/g/personal/yhao-zhang_stu_hit_edu_cn/Ev1ZhrDUVU1EmJHg81y1-eYBdMRRbzb1SpXxQJtodMGsfg?e=NfFFXI). We don't provide the multicoil dataset files since these files are too large. But you could find the the single-coil and **multi-coil** dataset pre-processing and creating code in [yhao-z/ocmr-preproc-tf](https://github.com/yhao-z/ocmr-preproc-tf). You may need to download the following files and put them in `./data` file folder.

```shell
# the data needs to be arranged into four sub file folders, and you may set the datadir in the code.
- train
	ocmr_train.tfrecord
- val
	ocmr_val.tfrecord
- test
	ocmr_test.tfrecord
- masks
    val_radial_16.npz
    test_radial_16.npz
```

## 2. Run the code

### Test only

We provide the training weights on **OCMR** dataset for all sampling cases that mentioned in our paper as in `weights-ocmr`. Note that the provided weights are only applicable in our data pre-processing implementation. **If you are using other different configuration, retraining from scratch is needed.**

```shell
# Please refer to main.py for more configurations.
python main.py --mode 'test'
```

### Training

```shell
# Please refer to main.py for more configurations.
python main.py --mode 'train'
```

## 3. Citation

If you find this work useful for your research, please cite:

```
@article{zhang2025jotlasnet,
  title={JotlasNet: Joint tensor low-rank and attention-based sparse unrolling network for accelerating dynamic MRI},
  author={Zhang, Yinghao and Gui, Haiyan and Yang, Ningdi and Hu, Yue},
  journal={Magnetic Resonance Imaging},
  pages={110337},
  year={2025},
  publisher={Elsevier}
}
```

