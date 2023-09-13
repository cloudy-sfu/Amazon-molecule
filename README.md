# GC ATTO PTRMS

Granger causality inference of PTR-MS measurement from ATTO project

![](https://shields.io/badge/dependencies-Python_3.11-blue?style=flat-square)

## Datasets

### ATTO database

**PTRMS**

Proton Transfer Reaction Mass Spectrometer (PTR-MS)

| Name      | Type    | Unit | Description                                                  |
| --------- | ------- | ---- | ------------------------------------------------------------ |
| timestamp | integer | s    | UNIX timestamp. Time zone is UTC-4, no daylight saving time. |
| M1\~M16   | real    | ppb  | Parts per billion of different molecular mass. The machine analysis the gas sample every 20 seconds, and this value is their average in 5 minutes. |
| height    | integer | m    | Altitude of where the machine locates.                       |

**mass**

| Name     | Type | Unit  | Description                                                  |
| -------- | ---- | ----- | ------------------------------------------------------------ |
| col_name | text |       | Column name in PTRMS table.                                  |
| mass     | real | g/mol | Molecular mass. The weighted mean of isotope, weighted by their portion in gas sample. |


**LOD_profile**

Limit of detection (profile 3.5min)

| Name           | Type    | Unit  | Description                                                  |
| -------------- | ------- | ----- | ------------------------------------------------------------ |
| mass           | real    | g/mol | Molecular mass.                                              |
| timestamp_from | integer | s     | UNIX timestamp. It corresponds to the start of applied limit of detection in `value` column. |
| timestamp_to   | integer | s     | UNIX timestamp. It corresponds to the end of applied limit of detection in `value` column. |
| value          | real    | ppb   | Limit of detection. It is the average in 3.5 minutes.        |

### Background information

Season:

|        |      |         |       |         |
| ------ | ---- | ------- | ----- | ------- |
| Month  | 1\~4 | 5\~6    | 7\~10 | 11\~12  |
| Season | wet  | wet-dry | dry   | dry-wet |

Location: S $2\degree 08.756'$, W $59 \degree 00.335'$

Project: [Amazon Tall Tower Observatory (ATTO)](https://attoproject.org)





Note:

- [x] Make a validation set.
- [ ] Multi-test statistic correction. Benjamini Hochberg correction
- [ ] Train again on different lengths of moving windows.
- [ ] Train again on different sampling frequencies.
- [ ] Validation method: Use the trained LSTM to predict other small time periods.

