# Amazon molecule

Granger causality inference of PTR-MS measurement from ATTO project

![](https://shields.io/badge/dependencies-Python_3.11-blue?style=flat-square)
![](https://shields.io/badge/dependencies-CUDA_11.8-green?style=flat-square)
![](https://shields.io/badge/dependencies-CuDNN_8.6.0-green?style=flat-square)
![](https://shields.io/badge/device-NVIDIA_GeForce_RTX_4090-green?style=flat-square)
![](https://shields.io/badge/OS-Ubuntu_22.04-lightgrey?style=flat-square)

## Devices

GPU should be more powerful than one of the following:

- 1 * NVIDIA GeForce RTX 4090
- 1 * NVIDIA A100 Tensor Core
- 1 * AWS p3.2xlarge EC2 instance

Minimum memory: 64GB physical + 32GB SWAP

Minimum graphic memory: 24GB

Recommended CPU: AMD Ryzen 9 3900X 12-Core Processor (or equivalently powerful)

With the above equipment, I recommend parallel 5 script instances of scripts that support parallel computing (described in "Usage"). It typically costs 40 hours on the given dataset.

## Acknowledgments

Rosol's [Nonlincausality](https://github.com/mrosol/Nonlincausality) is modified, under MIT license, copied to `method_rosol2022`.

## Usage

To replicate my experiments, please run scripts in order of the number prefix. 

Some scripts include the symbol `# %%`, which means a section in [JetBrains PyCharm Professional](https://www.jetbrains.com/pycharm/promo/), which is equivalent to a cell in Jupyter Notebook. Without special instructions below, users can run either one by one section or the whole script. The simplest way to run one by one section is to copy the code of a section to the Python interpreter, and to press 'Enter' to run. After one section is finished, run the next one. I don't use the Jupyter notebook version, because some users don't have a Jupyter notebook environment. They cannot read the notebook easily, but can still paste interactive code to the Python interpreter. It is not easy to use Google Colab, because data is from local machines instead of URLs on the Internet like Yahoo! Finance.

The following scripts support parallel computing acceleration. Users can run `python $script_name` within multiple `screen` instances. If their tasks are the same, each thread will skip the iterations in the main `for` loop, which another thread is working on. If a thread is killed, it has already saved the models they already trained, and can continue its task. If more threads join halfway, they can work on the remaining iterations in the main `for` loop and accelerate the overall progress.

```
3_causal_same_height.py
5_ur_boot_same_height.py
9_causal_same_molecule.py
11_ur_boot_same_molecule.py
```

Each section in script 2 is dependent on previous sections. It's necessary to run each section in an interactive environment, instead of running the whole script.

Repeat running script 3 with different combinations of constants as follows.

- dataset_name='default_15min', lag in (48, 96, 192)
- dataset_name='default_1h', lag in (12, 24, 48)
- dataset_name='default_4h', lag in (3, 6, 12)

It doesn't use `ArgumentParser` because the script consumes a long time and can be parallelled. So, it is not feasible to start running with multiple parameters using one batch script.

Repeat running script 7 with constants corresponding to script 3. This script takes about 5 minutes for one repeat; thus, it is feasible to use `ArgumentParser`. The example usage is:

```
python 7_wilcoxon_same_height.py --lag=48 --dataset_name='default_15min'
```

Do the same thing for script 8.

Repeat running script 14 with different combinations of constants as follows. Scripts 14, 15 are diagnosis scripts: users don't need to run every group of constants. Users only need to run constants where they suspect the numerical results to have problems.

- dataset_name='default_15min'
- lag=96
- n_boot_height=700, mode='same_height', height in ('h80', 'h150', 'h320'), j=0:12
- n_boot_molecule=400, mode='same_molecule', mass in ('M1', ..., 'M13'), j=0:2

The number of boots depends on the constant in script 12. The example usage is:

```
python 14_diagnose_ssr_resid_distribution.py --lag=96 --dataset_name='default_15min' --n_boot_height=700 --height='h150' --mode='same_height' --j=0
```

Correspondingly, run script 15 with different combinations of constants. It doesn't require `n_boot_height` or `n_boot_molecule`. The example usage is:

```
python 15_diagnose_resid_lld.py --lag=96 --dataset_name='default_15min' --height='h150' --mode='same_height' --j=0
```

## Background

Season: Month 1\~4 is the wet season, month 5\~6 is the wet-dry season, month 7\~10 is the dry season, and month 11\~12 is the dry-wet season.

Location: $S\ 2\degree 08.756', W\ 59 \degree 00.335'$

Project: [Amazon Tall Tower Observatory (ATTO)](https://attoproject.org)

## ATTO database

**PTRMS**

Proton Transfer Reaction Mass Spectrometer (PTR-MS)

| Name      | Type    | Unit | Description                                                  |
| --------- | ------- | ---- | ------------------------------------------------------------ |
| timestamp | integer | s    | UNIX timestamp. The time zone is UTC-4, with no daylight saving time. |
| M1\~M16   | real    | ppb  | Parts per billion of different molecular mass. The machine analyzes the gas sample every 20 seconds, and this value is their average in 5 minutes. |
| height    | integer | m    | Altitude of where the machine is located.                    |

**mass**

| Name     | Type | Unit  | Description                                                  |
| -------- | ---- | ----- | ------------------------------------------------------------ |
| col_name | text |       | Column name in PTRMS table.                                  |
| mass     | real | g/mol | Molecular mass. The weighted mean of isotope by their portion in the gas sample. |


**LOD_profile**

Limit of detection (profile 3.5min)

| Name           | Type    | Unit  | Description                                                  |
| -------------- | ------- | ----- | ------------------------------------------------------------ |
| mass           | real    | g/mol | Molecular mass.                                              |
| timestamp_from | integer | s     | UNIX timestamp. It corresponds to the start of the applied limit of detection in `value` column. |
| timestamp_to   | integer | s     | UNIX timestamp. It corresponds to the end of the applied limit of detection in `value` column. |
| value          | real    | ppb   | Limit of detection. It is an average of 3.5 minutes.         |

