# t5-dst-modified-pytorch
This is an unofficial implementation of the paper *Leveraging Slot Descriptions for Zero-Shot Cross-Domain Dialogue State Tracking*[[1]](#1).

Additionally, this repository also contains the modified version of *T5-dst*, which takes multiple slot types as a single input by exploiting the pre-training object of the original *T5*[[2]](#2) model.

The first image describes the original *T5-dst*'s procedure, using slot descriptions.

<img src="https://user-images.githubusercontent.com/16731987/133780757-582fbece-8754-486e-9328-b7729f8067f8.png" alt="The description of original T5-dst."/>

<br/>

And the second shows the modified version of *T5-dst*, inspired by *T5*'s pre-training object.

<img src="https://user-images.githubusercontent.com/16731987/133783669-6628f87f-6e76-4c36-994c-2e6f87d2159c.png" alt="The description of T5-dst modified."/>

<br/>

---

### Dataset

This project uses MultiWoZ 2.1[[3]](#3), which can be downloaded [here](https://github.com/budzianowski/multiwoz/blob/master/data/MultiWOZ_2.1.zip).

Download the zip file, unzip it, and put the whole `"MultiWOZ_2.1"` directory into `"data/raw"`.

If you do not have the directory, make `"data/raw"` directory yourself.

The structure of `"data"` directory is as follows.

```
data
    └--raw
        └--MultiWOZ_2.1
            └--attraction_db.json
            └--data.json
            └--hospital_db.json
            └--...
            └--valListFile.txt
```

<br/>

---

### Arguments

**Arguments for data processing**

| Argument              | Type  | Description                                      | Default        |
| --------------------- | ----- | ------------------------------------------------ | -------------- |
| `--data_dir`          | `str` | The root directory for the entire data files.    | `"data"`       |
| `--raw_dir`           | `str` | The directory which contains raw data files.     | `"raw"`        |
| `--cached_dir`        | `str` | The directory for cached files after processing. | `"cached"`     |
| `--train_prefix`      | `str` | The train data prefix.                           | `"train"`      |
| `--valid_prefix`      | `str` | The validation data prefix.                      | `"valid"`      |
| `--test_prefix`       | `str` | The test data prefix.                            | `"test"`       |
| `--slot_descs_prefix` | `str` | The slot description file prefix.                | `"slot_descs"` |

<br/>

**Arguments for basic training**

The basic training is for the original *T5-dst* setting.

| Argument              | Type         | Description                                                  | Default             |
| --------------------- | ------------ | ------------------------------------------------------------ | ------------------- |
| `--seed`              | `int`        | The random seed.                                             | `0`                 |
| `--data_dir`          | `str`        | The root directory for the entire data files.                | `"data"`            |
| `--cached_dir`        | `str`        | The directory for cached files after processing.             | `"cached"`          |
| `--data_name`         | `str`        | The data name to train/evaluate. (`"multiwoz_fullshot"` or `"multiwoz_zeroshot"`) | *YOU MUST SPECIFY*  |
| `--trg_domain`        | `str`        | The target domain to be excluded in zero-shot setting. (`"attraction"` or `"hotel"` or `"restaurant"` or `"taxi"` or `"train"`) | *YOU MIGHT SPECIFY* |
| `--model_name`        | `str`        | The T5 model type. (`"t5-small"` or `"t5-base"`)             | `"t5-small"`        |
| `--train_prefix`      | `str`        | The train data prefix.                                       | `"train"`           |
| `--valid_prefix`      | `str`        | The validation data prefix.                                  | `"valid"`           |
| `--test_prefix`       | `str`        | The test data prefix.                                        | `"test"`            |
| `--slot_descs_prefix` | `str`        | The slot description file prefix.                            | `"slot_descs"`      |
| `--num_epochs`        | `str`        | The total number of training epochs.                         | `10`                |
| `--train_batch_size`  | `int`        | The batch size for train data loader.                        | `32`                |
| `--eval_batch_size`   | `int`        | The batch size for evaluation data loader.                   | `8`                 |
| `--num_workers`       | `int`        | The number of subprocesses for data loading.                 | `0`                 |
| `--src_max_len`       | `int`        | The maximum length of the source sequence.                   | `512`               |
| `--trg_max_len`       | `int`        | The maximum length of the target sequence.                   | `128`               |
| `--learning_rate`     | `float`      | The initial learning rate.                                   | `1e-4`              |
| `--warmup_ratio`      | `float`      | The ratio of warmup steps to total training steps.           | `0.0`               |
| `--max_grad_norm`     | `float`      | The maximum value of gradient.                               | `1.0`               |
| `--min_delta`         | `float`      | The minimum delta value for evaluation metric.               | `1e-4`              |
| `--patience`          | `int`        | The number patience epochs before early stopping.            | `3`                 |
| `--sep_token`         | `str`        | The special token for separation.                            | `"<sep>"`           |
| `--gpu`               | `str`        | The indices of GPUs. (ex: `"0, 1"`)                          | `"0"`               |
| `--log_dir`           | `str`        | The location of lightning log directory.                     | `"./"`              |
| `--use_cached`        | `store_true` | Using cached data or not?                                    | -                   |

<br/>

**Arguments for modified training**

The basic training is for the modified *T5-dst* setting.

| Argument              | Type         | Description                                                  | Default             |
| --------------------- | ------------ | ------------------------------------------------------------ | ------------------- |
| `--seed`              | `int`        | The random seed.                                             | `0`                 |
| `--data_dir`          | `str`        | The root directory for the entire data files.                | `"data"`            |
| `--cached_dir`        | `str`        | The directory for cached files after processing.             | `"cached"`          |
| `--data_name`         | `str`        | The data name to train/evaluate. (`"multiwoz_fullshot"` or `"multiwoz_zeroshot"`) | *YOU MUST SPECIFY*  |
| `--trg_domain`        | `str`        | The target domain to be excluded in zero-shot setting. (`"attraction"` or `"hotel"` or `"restaurant"` or `"taxi"` or `"train"`) | *YOU MIGHT SPECIFY* |
| `--model_name`        | `str`        | The T5 model type. (`"t5-small"` or `"t5-base"`)             | `"t5-small"`        |
| `--train_prefix`      | `str`        | The train data prefix.                                       | `"train"`           |
| `--valid_prefix`      | `str`        | The validation data prefix.                                  | `"valid"`           |
| `--test_prefix`       | `str`        | The test data prefix.                                        | `"test"`            |
| `--slot_descs_prefix` | `str`        | The slot description file prefix.                            | `"slot_descs"`      |
| `--num_epochs`        | `str`        | The total number of training epochs.                         | `10`                |
| `--train_batch_size`  | `int`        | The batch size for train data loader.                        | `32`                |
| `--eval_batch_size`   | `int`        | The batch size for evaluation data loader.                   | `8`                 |
| `--num_workers`       | `int`        | The number of subprocesses for data loading.                 | `0`                 |
| `--src_max_len`       | `int`        | The maximum length of the source sequence.                   | `512`               |
| `--trg_max_len`       | `int`        | The maximum length of the target sequence.                   | `128`               |
| `--max_extras`        | `int`        | The maximum number of slot types to include in one input.    | `5`                 |
| `--learning_rate`     | `float`      | The initial learning rate.                                   | `1e-4`              |
| `--warmup_ratio`      | `float`      | The ratio of warmup steps to total training steps.           | `0.0`               |
| `--max_grad_norm`     | `float`      | The maximum value of gradient.                               | `1.0`               |
| `--min_delta`         | `float`      | The minimum delta value for evaluation metric.               | `1e-4`              |
| `--patience`          | `int`        | The number patience epochs before early stopping.            | `3`                 |
| `--sep_token`         | `str`        | The special token for separation.                            | `"<sep>"`           |
| `--gpu`               | `str`        | The indices of GPUs. (ex: `"0, 1"`)                          | `"0"`               |
| `--log_dir`           | `str`        | The location of lightning log directory.                     | `"./"`              |
| `--use_cached`        | `store_true` | Using cached data or not?                                    | -                   |

<br/>

---

### How to run

1. Install all required packages.

   ```shell
   pip install -r requirements.txt
   ```

   <br/>

2. Parse & precess the raw data. Make sure that raw data files are in the appropriate location.

   ```shell
   sh exec_data_process.sh
   ```

   After running this, you will have following files in `data` folder.

   ```
   data
       └--raw
           └--MultiWOZ_2.1
               └--attraction_db.json
               └--data.json
               └--hospital_db.json
               └--...
               └--valListFile.txt
       └--cached
           └--multiwoz_fullshot
               └--train_slot_descs.json
               └--train_utters.pickle
               └--train_states.json
               └--valid_slot_descs.json
               └--valid_utters.pickle
               └--valid_states.json
               └--test_slot_descs.json
               └--test_utters.pickle
               └--test_states.json
           └--multiwoz_zeroshot
               └--attraction
                   └--train_slot_descs.json
                   └--train_utters.pickle
                   └--train_states.json
                   └--valid_slot_descs.json
                   └--valid_utters.pickle
                   └--valid_states.json
                   └--test_slot_descs.json
                   └--test_utters.pickle
                   └--test_states.json
               └--hotel
                   └--...
               └--restaurant
                   └--...
               └--taxi
                   └--...
               └--train
                   └--...
   ```

   <br/>

3. Now you can run the scripts for training. This project provides 4 different shell script files for each training setting. One thing you should keep in mind is that default argument values are set based on full-shot training, not zero-shot condition. In script files for zero-shot training, each argument value might be different from default value indicated in above tables.

   Additionally, the target domain is not provided at first. You should specify the target domain in `exec_zeroshot_basic.sh` or `exec_zeroshot_modified.sh`.

   ```shell
   sh exec_fullshot_basic.sh  # Full-shot + Basic
   sh exec_fullshot_modified.sh  # Full-shot + Modified
   sh exec_zeroshot_basic.sh  # Zero-shot + Basic
   sh exec_zeroshot_modified.sh  # Zero-shot + Modified
   ```

   For the first running, pre-processed input & outputs files for train, validation, test are saved. Therefore, from the second run, by adding `--use_cached` argument in the shell script file, you can directly run the codes without redundant pre-processing.

<br/>

---

### References

<a id="1">[1]</a> Lin, Z., Liu, B., Moon, S., Crook, P., Zhou, Z., Wang, Z., ... & Subba, R. (2021). Leveraging Slot Descriptions for Zero-Shot Cross-Domain Dialogue State Tracking. *arXiv preprint arXiv:2105.04222*. ([https://arxiv.org/pdf/2105.04222.pdf](https://arxiv.org/pdf/2105.04222.pdf))

<a id="2">[2]</a> Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., ... & Liu, P. J. (2019). Exploring the limits of transfer learning with a unified text-to-text transformer. *arXiv preprint arXiv:1910.10683*. ([https://www.jmlr.org/papers/volume21/20-074/20-074.pdf](https://www.jmlr.org/papers/volume21/20-074/20-074.pdf))

<a id="3">[3]</a> Eric, M., Goel, R., Paul, S., Kumar, A., Sethi, A., Ku, P., ... & Hakkani-Tur, D. (2019). MultiWOZ 2.1: A consolidated multi-domain dialogue dataset with state corrections and state tracking baselines. *arXiv preprint arXiv:1907.01669*. ([https://arxiv.org/pdf/1907.01669.pdf](https://arxiv.org/pdf/1907.01669.pdf))
