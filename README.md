# AI-IMU Dead-Reckoning [[IEEE paper](https://ieeexplore.ieee.org/document/9035481), [ArXiv paper](https://arxiv.org/pdf/1904.06064.pdf)]

see [original repo](https://github.com/mbrossar/ai-imu-dr) for more details.


## Code
Our implementation is done in Python. We use [Pytorch](https://pytorch.org/) for the adapter block of the system. The code was tested under Python 3.5.

### Installation & Prerequies
1.  Prepare docker environment via [DevUtils](https://github.com/deepmirrorinc/DevUtils)
```
dev
```


### Test Dm Model

Train a model to predict the covariance of velocity of xyz, trained directly by gt measurement.

```
cd src
python train_dm.py
```

```
python main_dm_dm.py
```

### Test Original Model

see [original repo](https://github.com/mbrossar/ai-imu-dr) for more KITTI traing, and its original implementation.

**Test in DM car data** (using iterative EKF, with the same measurement update as the project):

| with model | without model |
|---|---|
| ![Screenshot from 2023-12-27 15-47-38](https://github.com/yeliu-deepmirror/ai-imu-dr/assets/74998488/0228a8fd-b3dc-402c-a812-d81672543ef0) | ![Screenshot from 2023-12-27 15-47-51](https://github.com/yeliu-deepmirror/ai-imu-dr/assets/74998488/65fa3b58-997e-477a-9d2e-8162f72de88d) |


## Paper
The paper M. Brossard, A. Barrau and S. Bonnabel, "AI-IMU Dead-Reckoning," in _IEEE Transactions on Intelligent Vehicles_, 2020, relative to this repo is available at this [url](https://cloud.mines-paristech.fr/index.php/s/8YDqD0Y1e6BWzCG).

### Citation

If you use this code in your research, please cite:

```
@article{brossard2019aiimu,
  author = {Martin Brossard and Axel Barrau and Silv\`ere Bonnabel},
  journal={IEEE Transactions on Intelligent Vehicles},
  title = {{AI-IMU Dead-Reckoning}},
  year = {2020}
}
```

### Authors
Martin Brossard*, Axel Barrau° and Silvère Bonnabel*

*MINES ParisTech, PSL Research University, Centre for Robotics, 60 Boulevard Saint-Michel, 75006 Paris, France

°Safran Tech, Groupe Safran, Rue des Jeunes Bois-Châteaufort, 78772, Magny Les Hameaux Cedex, France
