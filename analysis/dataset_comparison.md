## Comparing MerRec to Other Large Scale E-Commerce Datasets

|         Dataset         |  Market Type  |  Users  |  Items  | Interactions |                              Interaction Type                              | Categories / Leveled? | Brand | Price | Color | Size  | Timestamp | Item Tokens | SKU/UPC | Covered Year | Sessions |
| :---------------------: | :-----------: | :-----: | :-----: | :----------: | :------------------------------------------------------------------------: | :-------------------: | :---: | :---: | :---: | :---: | :-------: | :---------: | :-----: | :----------: | :------: |
|      MerRec (2024)      | C2C (General) |  5.56M  | 83.07M  |    1.27B     | Click, Like, Add-to-cart, Make Offer, Initiate Purchase, Complete Purchase |      3399 / Yes       |  Yes  |  Yes  |  Yes  |  Yes  |    Yes    |   18.86B    |   No    | Half of 2023 | 227.16M  |
|      Amazon (2023)      | B2C (General) | 54.51M  | 48.19M  |   571.54M    |                               Rating, Review                               |        33 / No        |  Yes  |  Yes  |  Yes  |  Yes  |    Yes    |   30.78B    |   Yes   |  1996-2023   |    No    |
|      Tmall (2016)       | B2C (General) | 645.37K |  2.35M  |    44.52M    |                              Click, Purchase                               |        72 / No        |  No   |  No   |  No   |  No   |    Yes    |     No      |   Yes   | Half of 2015 | 200.28K  |
|    Amazon-M2 (2023)     | B2C (General) |   No    |  1.41M  |    16.79M    |                                     No                                     |        No / No        |  Yes  |  No   |  Yes  |  Yes  |    No     |     Yes     |   Yes   |      ?       |  3.96M   |
|    DIGINETICA (2016)    |       ?       | 232.93K | 184.04K |     3.3M     |                       View, Click, Purchase, Search                        |       1217 / No       |  No   |  Yes  |  No   |  No   |    Yes    |   941.64K   |    ?    |      ?       | 573.93K  |
|    YOOCHOOSE (2015)     |       ?       |   No    | 52.73K  |    34.15M    |                                  Purchase                                  |       348 / No        |  No   |  Yes  |  No   |  No   |    Yes    |     No      |    ?    |      ?       |  9.24M   |
|   Retailrocket (2022)   |       ?       |  1.40M  | 417.05K |    2.75M     |                        View, Add-to-cart, Purchase                         |      1669 / Yes       |  No   |  No   |  No   |  No   |    Yes    |   51.29M    |    ?    |      ?       |    No    |
|     Ali-CCP (2018)      | C2C (General) |  400K   |  4.3M   |    87.41M    |                           View, Click, Purchase                            |         ? / ?         |  Yes  |  No   |  No   |  No   |    No     |     No      |    ?    |      ?       |    No    |
| Alibaba-iFashion (2019) | C2C (Fashion) |  3.56M  |  4.46M  |   191.39M    |                                   Click                                    |        75 / No        |  No   |  No   |  No   |  No   |    No     |    7.7M     |    ?    |      ?       |    No    |

## Notes

- To our best awareness, only MerRec and Retailrocket [6] have item snapshots.
- Some columns (e.g. condition, shipment paying party) are omitted here to prevent the table from being too large.
- In practice, token estimates for DIGINETICA [4] and Retailrocket [6] should be higher than numbers shown here, as the stated numbers here represent counting over unique snapshots rather than all (possibly redundant) snapshots. MerRec estimate contains all snapshot instances.
- The Amazon (2023) [1] review dataset is not a conventional action event based recommendation dataset and cannot be used interchangeably with MerRec. These two do not generally conform to the same machine learning tasks.
- Retailrocket [6] hashed most of its item metadata field names, so it is unclear if it contains some of the metadata compared here.
- For Ali-CCP [7], we interpreted "impression" as "view", and "conversion" as "purchase".
- For Alibaba-iFashion [8], the tokens provided were mostly Chinese characters. If we define tokens as the number of Chinese characters (as well as other occasional English characters and numerics) instances instead of splitting by empty space, then the token count would be 164.18M.

## Reference

Note: All datasets below are cited as their most-recent version available. References are provided as Bibtex if available to be format agnostic.

[1] Amazon (2023): https://amazon-reviews-2023.github.io/
```tex
@article{hou2024bridging,
  title={Bridging Language and Items for Retrieval and Recommendation},
  author={Hou, Yupeng and Li, Jiacheng and He, Zhankui and Yan, An and Chen, Xiusi and McAuley, Julian},
  journal={arXiv preprint arXiv:2403.03952},
  year={2024}
}
```

[2] Tmall (2016): https://tianchi.aliyun.com/dataset/53
```tex
@misc{tmall2016,
    title={IJCAI-16 Brick-and-Mortar Store Recommendation Dataset},
    url={https://tianchi.aliyun.com/dataset/dataDetail?dataId=53},
    author={Tianchi},
    year={2018}
}
```

[3] Amazon-M2 (2023): https://kddcup23.github.io/
```tex
@inproceedings{jin2023amazonm,
  title={Amazon-M2: A Multilingual Multi-locale Shopping Session Dataset for Recommendation and Text Generation},
  author={Wei Jin and Haitao Mao and Zheng Li and Haoming Jiang and Chen Luo and Hongzhi Wen and Haoyu Han and Hanqing Lu and Zhengyang Wang and Ruirui Li and Zhen Li and Monica Xiao Cheng and Rahul Goutam and Haiyang Zhang and Karthik Subbian and Suhang Wang and Yizhou Sun and Jiliang Tang and Bing Yin and Xianfeng Tang},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2023},
  url={https://openreview.net/forum?id=uXBO47JcJT}
}
```

[4] DIGINETICA (2016):
- https://competitions.codalab.org/competitions/11161#learn_the_details-data2

[5] YOOCHOOSE (2015):
- https://recsys.acm.org/recsys15/challenge/

[6] Retail Rocket (2022): https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset
```tex
@misc{zykov2022retailrocket,
	title={Retailrocket recommender system dataset},
	url={https://www.kaggle.com/dsv/4471234},
	DOI={10.34740/KAGGLE/DSV/4471234},
	publisher={Kaggle},
	author={Roman Zykov and Noskov Artem and Anokhin Alexander},
	year={2022}
}
```

[7] Ali-CCP (2018): https://tianchi.aliyun.com/dataset/408
```tex
@inproceedings{ma2018esmm,
  title={Entire space multi-task model: An effective approach for estimating post-click conversion rate},
  author={Ma, Xiao and Zhao, Liqin and Huang, Guan and Wang, Zhi and Hu, Zelin and Zhu, Xiaoqiang and Gai, Kun},
  booktitle={The 41st International ACM SIGIR Conference on Research \& Development in Information Retrieval},
  pages={1137--1140},
  year={2018}
}
```

[8] Alibaba-iFashion (2019): https://github.com/wenyuer/POG?tab=readme-ov-file
```tex
@inproceedings{chen2019pog,
  author = {Chen, Wen and Huang, Pipei and Xu, Jiaming and Guo, Xin and Guo, Cheng and Sun, Fei and Li, Chao and Pfadler, Andreas and Zhao, Huan and Zhao, Binqiang},
  title = {POG: Personalized Outfit Generation for Fashion Recommendation at Alibaba iFashion},
  year = {2019},
  isbn = {9781450362016},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3292500.3330652},
  doi = {10.1145/3292500.3330652},
  booktitle = {Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  pages = {2662–2670},
  numpages = {9},
  keywords = {transformer, self-attention, fashion outfit recommendation, fashion outfit generation, deep learning},
  location = {Anchorage, AK, USA},
  series = {KDD '19}
}
```