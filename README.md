# CHORE (ECCV'22)
#### Official implementation of the ECCV 2022 paper: Contact, Human and Object REconstruction from a single RGB image
[[ArXiv]](https://arxiv.org/abs/2204.02445) [[Project Page]](http://virtualhumans.mpi-inf.mpg.de/chore)
<p align="left">
<img src="https://virtualhumans.mpi-inf.mpg.de/chore/teaser.gif" alt="teaser" width="512"/>
</p>


## Contents
1. [Dependencies](#dependencies)
2. [Run demo](#run-demo)
3. [Training](#training)
4. [Testing](#testing)
5. [License](#license)
6. [Citation](#citation)


## Dependencies
The code is tested with `torch 1.6, cuda10.1, debian 11`.  We recommend using anaconda environment: 
```shell
conda create -n chore python=3.7
conda activate chore 
```
Main dependencies are listed in `requirements.txt`, install them with 
```shell
git clone https://github.com/xiexh20/CHORE.git && cd CHORE 
pip install -r requirements.txt
```

Installing other dependencies:
1. psbody-mesh library. see [installation](https://github.com/MPI-IS/mesh#installation).
2. [igl library](https://libigl.github.io/libigl-python-bindings/). `conda install -c conda-forge igl`
3. [Detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) library: 
```shell
python -m pip install detectron2 -f   https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.6/index.html
```
4. [Neural Mesh Renderer](https://github.com/JiangWenPL/multiperson/tree/master/neural_renderer): `pip install external/neural_renderer`
5. [Mesh intersection library](https://github.com/vchoutas/torch-mesh-isect): 
```shell
export CHORE_PATH=${PWD}
git clone https://github.com/NVIDIA/cuda-samples.git external/cuda-samples
export CUDA_SAMPLES_INC=${CHORE_PATH}/external/cuda-samples/Common/
git clone https://github.com/vchoutas/torch-mesh-isect external/torch-mesh-isect
cp external/torch-mesh-isect/include/double_vec_ops.h external/torch-mesh-isect/src/
```
Add these lines to `external/torch-mesh-isect/src/bvh.cpp` before `AT_CHECK` is defined ([reference](    https://github.com/vchoutas/torch-mesh-isect/issues/23)):
```cpp
#ifndef AT_CHECK 
#define AT_CHECK TORCH_CHECK 
#endif 
```
finally run `pip install external/torch-mesh-isect/`

## Run demo
Pretrained model can be downloaded from [here](https://nextcloud.mpi-klsb.mpg.de/index.php/s/SatwEeqFnQdBGaF). Please download and extract it in the code directory with: `unzip chore-pretrained.zip -d experiments`

We use the SMPL-H body model, please prepare it from the [official website](https://mano.is.tue.mpg.de/) and modify `SMPL_MODEL_ROOT` in `PATHS.yml` accordingly.

Run demo example with: 
```shell
python demo.py chore-release -s example -on basketball 
```
results will be saved to  `example/000000117377/demo`.
## Training 
Please follow the instructions [here](http://virtualhumans.mpi-inf.mpg.de/behave/license.html) to download the [BEHAVE dataset](http://virtualhumans.mpi-inf.mpg.de/behave/).

After download and unzip, modify the `BEHAVE_PATH` in file `PATHS.yml` accordingly. 
#### Data preprocessing
Specify the root path `PROCESSED_PATH` in file `PATHS.yml` to where you want the processed data to be saved. 

Preprocess one sequence:
```shell
python preprocess/preprocess_scale.py -s [path to one sequence]
```
Alternatively you can run `python preprocess/preprocess_scale.py -a` to process all sequences sequentially. 

#### Train the model 
Specify the number of GPUs by `nproc_per_node` and run the following to start training:
```shell
python -m torch.distributed.launch --nproc_per_node=4 --use_env train_launch.py -en chore-release
```
## Testing
For BEHAVE dataset, we test on images from kinect one (k1) and evaluate on images where object is occluded less than 70%. To compute the occlusion ratio, you need to render full object masks. 

Our rendered full object masks can be downloaded [here](https://datasets.d2.mpi-inf.mpg.de/cvpr22behave/behave-test-object-fullmask.zip). Download and extract them to the *same path* where BEHAVE sequences were extracted.
 

#### Test on BEHAVE test set
After downloading the pretrained model, you can test one sequence from behave data with:
```shell
python recon/recon_fit_behave.py chore-release --save_name chore-release -s [path to one sequence]
```

#### Evaluate 
Run the following to compute errors reported in table 1: 
```shell
python recon/evaluate.py 
```

## License
Copyright (c) 2022 Xianghui Xie, Max-Planck-Gesellschaft

Please read carefully the following terms and conditions and any accompanying documentation before you download and/or use this software and associated documentation files (the "Software").

The authors hereby grant you a non-exclusive, non-transferable, free of charge right to copy, modify, merge, publish, distribute, and sublicense the Software for the sole purpose of performing non-commercial scientific research, non-commercial education, or non-commercial artistic projects.

Any other use, in particular any use for commercial purposes, is prohibited. This includes, without limitation, incorporation in a commercial product, use in a commercial service, or production of other artefacts for commercial purposes.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

You understand and agree that the authors are under no obligation to provide either maintenance services, update services, notices of latent defects, or corrections of defects with regard to the Software. The authors nevertheless reserve the right to update, modify, or discontinue the Software at any time.

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. You agree to cite the **CHORE: Contact, Human and Object REconstruction from a single RGB image** paper in documents and papers that report on research using this Software.


## Citation
If you use our code, please cite:
```bibtex
@inproceedings{xie2022chore,
    title = {CHORE: Contact, Human and Object REconstruction from a single RGB image},
    author = {Xie, Xianghui and Bhatnagar, Bharat Lal and Pons-Moll, Gerard},
    booktitle = {ECCV},
    month = {October},
    year = {2022},
}
```
If you use BEHAVE dataset, please also cite:
```bibtex
@inproceedings{bhatnagar22behave,
    title = {BEHAVE: Dataset and Method for Tracking Human Object Interactions},
    author={Bhatnagar, Bharat Lal and Xie, Xianghui and Petrov, Ilya and Sminchisescu, Cristian and Theobalt, Christian and Pons-Moll, Gerard},
    booktitle = {{IEEE} Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {jun},
    organization = {{IEEE}},
    year = {2022},
    }
```




