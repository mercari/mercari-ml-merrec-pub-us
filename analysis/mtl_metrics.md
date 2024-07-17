# Multi-task learning (MTL) for Recommendation

Below we added 4 additional stronger baselines.

|      Model       | View AUC | Like AUC | Log Loss (View + Like) | Train+Val Time (Hrs) | VRAM (GB) |
| :--------------: | :------: | :------: | :--------------------: | :------------------: | :-------: |
| MMOE (Only view) |  0.709   |   N/A    |         0.395          |         16.3         |   3.93    |
| MMOE (Only like) |   N/A    |  0.709   |         0.354          |         16.6         |   3.93    |
|       MMOE       |  0.712   |  0.713   |         0.744          |        22.61         |   3.94    |
|       ESMM       |  0.713   |  0.715   |         0.744          |         23.7         |   3.94    |
|       AITM       |  0.773   |  0.736   |         0.684          |         2.85         |   3.26    |
|       PLE        |  0.773   |  0.736   |         0.686          |         4.38         |   3.39    |
|       OMoE       |  0.773   |  0.737   |         0.684          |         3.3          |   4.17    |
|       STEM       |  0.772   |  0.736   |         0.683          |         9.68         |   11.27   |

## Notes

- For AITM [2], PLE [3], OMoE [4] and STEM [1], the AUC numbers were more different in the subsequent but truncated digits, though the margin is small and we do not observe any significant difference.
- Just like originally indicated in the paper, no model in this MTL experiment utilizes the text features, as this was the commonly-adopted standard way of evaluation in the literature.
- These 4 models were selected based on recent studies showing that they are at SoTA level.
- Implementation will be publicly released in the coming weeks. Some cleaning and organization is needed to accommodate these 4 models.
- The ME-MMOE and ME-PLE competitive baselines introduced by the STEM [1] paper were omitted here as some of the design details and implementation were not published by them.

## Reference

References here are presented as Bibtex entries to be format agnostic.

[1] STEM (AAAI 2024): https://ojs.aaai.org/index.php/AAAI/article/view/28749/
```tex
@article{su2024stem,
  author = {Liangcai Su and Junwei Pan and Ximei Wang and Xi Xiao and Shijie Quan and Xihua Chen and Jie Jiang},
  title  = {STEM: Unleashing the Power of Embeddings for Multi-task Recommendation},
  journal = {Proceedings of the 38-th AAAI Conference on Artificial Intelligence (AAAI 2024)},
  year    = {2024},
}
```

[2] AITM (KDD 2021): https://dl.acm.org/doi/10.1145/3447548.3467071
```tex
@inproceedings{xi2021aitm,
    author = {Xi, Dongbo and Chen, Zhen and Yan, Peng and Zhang, Yinger and Zhu, Yongchun and Zhuang, Fuzhen and Chen, Yu},
    title = {Modeling the Sequential Dependence among Audience Multi-step Conversions with Multi-task Learning in Targeted Display Advertising},
    year = {2021},
    isbn = {9781450383325},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3447548.3467071},
    doi = {10.1145/3447548.3467071},
    booktitle = {Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery \& Data Mining},
    pages = {3745–3755},
    numpages = {11},
    keywords = {targeted display advertising, sequential dependence, multi-task learning, multi-step conversions},
    location = {Virtual Event, Singapore},
    series = {KDD '21}
}
```

[3] PLE (RecSys 2020): https://dl.acm.org/doi/10.1145/3383313.3412236
```tex
@inproceedings{tang2020ple,
    author = {Tang, Hongyan and Liu, Junning and Zhao, Ming and Gong, Xudong},
    title = {Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations},
    year = {2020},
    isbn = {9781450375832},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3383313.3412236},
    doi = {10.1145/3383313.3412236},
    booktitle = {Proceedings of the 14th ACM Conference on Recommender Systems},
    pages = {269–278},
    numpages = {10},
    keywords = {Multi-task Learning, Recommender System, Seesaw Phenomenon},
    location = {Virtual Event, Brazil},
    series = {RecSys '20}
}
```

[4] OMoE (KDD 2018): https://dl.acm.org/doi/10.1145/3219819.3220007
```tex
@inproceedings{10.1145/3219819.3220007,
author = {Ma, Jiaqi and Zhao, Zhe and Yi, Xinyang and Chen, Jilin and Hong, Lichan and Chi, Ed H.},
title = {Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts},
year = {2018},
isbn = {9781450355520},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3219819.3220007},
doi = {10.1145/3219819.3220007},
booktitle = {Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
pages = {1930–1939},
numpages = {10},
keywords = {mixture of experts, multi-task learning, neural network, recommendation system},
location = {London, United Kingdom},
series = {KDD '18}
}
```
