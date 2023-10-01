# GC ATTO PTRMS

Granger causality inference of PTR-MS measurement from ATTO project

![](https://shields.io/badge/dependencies-Python_3.11-blue?style=flat-square)
![](https://shields.io/badge/hardware-NVIDIA_GeForce_RTX_4090-green?style=flat-square)
![](https://shields.io/badge/OS-Ubuntu_22.04-lightgrey?style=flat-square)

## Hardware requirements

GPU is more powerful than one of the following:

- 1 * NVIDIA GeForce RTX 4090
- 1 * NVIDIA A100 Tensor Core
- 1 * AWS p3.2xlarge EC2 instance (Oct 2023)

Minimum memory: 64GB physical + 32GB SWAP

Minimum graphic memory: 24GB

Recommended CPU is as powerful as: AMD Ryzen 9 3900X 12-Core Processor

With the above equipment, I recommend parallel 5 script instances such as `5_ur_boot_same_height.py` . It typically costs 40 hours on the given dataset.

## Acknowledgments

Rosol's [Nonlincausality](https://github.com/mrosol/Nonlincausality) is modified, under MIT license, copied to `method_rosol2022`.

## Usage

The following scripts support parallel computing acceleration. Users can run `python $script_name` within multiple `screen` instances. If their tasks are the same, each thread will skip the iterations in the main `for` loop, which another thread is working on. If a thread is killed, it has already saved the models they already trained, and can continue its task. If more threads join halfway, they can work on the remaining iterations in the main `for` loop and accelerate the overall progress.

```
3_causal_same_height.py
5_ur_boot_same_height.py
9_causal_same_molecule.py
11_ur_boot_same_molecule.py
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

