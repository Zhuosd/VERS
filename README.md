# VERS: Towards Voice-Enriched Recommendation Systems
## Abstract
Recommender Systems (RS) have become an important emerging research direction in the field of personalized user experiences, with their innate advantage of capturing user preferences through interactive dialogue and unveiling the reasons behind recommendations. However, most of the current RS are text-based, which can be less user-friendly and may pose significant challenges to users with visual impairments or limited writing abilities, while also neglecting the rich information present in voice inputs. To address this issue, this paper proposes a novel recommendation systems framework named Voice-Enriched Recommendation systems (VERS). This framework captures the intrinsic correlation between voice and text information, ensuring the consistency of user preference information representation. To better encode the voice characteristics of users, VERS establishes a transit station between voice and text feature information through feature retrieval. Then, by integrating all representation information of both user and item, we utilize higher-order graph convolutional networks to learn user-item interaction patterns, facilitating the prediction of user preferences. Experimental results on three real-world datasets are recorded to prove the feasibility and effectiveness of our proposed method.
## Requirements

```
conda create -n VERS python=3.9
conda activate VERS
pip install -r requirements.txt
```

## Datasets
You can get the audio datasets from [GoogleDrive](https://drive.google.com/file/d/1FnpYhMaeskckxGheKjar0U4YHIdDKM6K/view). Please extract the data under the `./data/{dataset_name}/mp3/`

## How to run VERS

### MFE

- Coat

```
python run_MFE.py --dataset=coat --epochs=35 --lr=0.0005 --test=1 --neg_num=3 
```

- Movielens-1m

```
python run_MFE.py --dataset=movielens1m --epochs=35 --lr=0.0005 --test=1 --neg_num=5 
```

### AGIP

- Coat

```
python main.py --weight_decay=1e-4 --lr=0.001 --n_layers=3 --dataset=coat --recdim=64 --neg_num 3 --ens_ratio 0.8
```

- Movielens-1m

```
python main.py --weight_decay=1e-4 --lr=0.001 --n_layers=3 --dataset=movielens1m --recdim=64 --neg_num 5 --ens_ratio 0.6
```



## Benchmarking

Coat:
|   Metrics   | VERS |
| ----------- | ----------- |
|  F1@10   |    **12.68**   |
|  Precision@10   |    **8.24**   |
| Recall@10   |    **27.47**   |
|  NDCG@10    |    **18.68**   |

[//]: # (## Citation)

[//]: # ()
[//]: # (You are welcome to cite our paper:)

[//]: # (```)

[//]: # ()
[//]: # (```)
