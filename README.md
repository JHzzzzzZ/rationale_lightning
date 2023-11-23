# rationale_lightning

## Introduction
A reimplementation of common rationale extraction methods based on `pytorch lightning`.
<!-- 基于pytorch lightning，复现常见rationale extraction的方法。 -->

## preparation
You should put your data under the `./data` directory. For example, you should put `BeerReview` dataset under the `./data/beer`.

Also, you should put pretrained word embeddings under `./pretrained`.

Maybe, in the `BeerReview` dataset, you need to unzip the `*.gz` file to `*.txt` file.

## run codes
For example: if you want to run the model on `BeerReview` dataset `apearance` aspect, you should run the below code.
```bash
bash run_binarized_beer_0.sh
```