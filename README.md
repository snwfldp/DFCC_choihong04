# DFCC (DeepFake speech Classification Challenge) choihong04
INU 25-1 Machine Learning DFCC Challenge Team choihong04

## Directory structure

```
.
├── 2501ml_data             # dataset
│   ├── label
│   ├── test
│   └── train
|
├── do_not_open             # 사용 X
│   └── do_not.ipynb
|
├── etc                     # 실행 코드 외 파일
│   ├── eval.pl             # 평가 코드
│   └── team_test_result.txt
|
├── model.pkl               # 모델
├── model.py                # 전처리 함수, 모델 설계
├── requirements.txt        # 필요 라이브러리 목록
├── test.ipynb              # test 코드
└── train.ipynb             # train 코드
```

## How to run

1. **Required libraries installing**
    ```sh
    pip install -r requirements.txt
    ```

2. **Train model**
    - Run `./train.ipynb`.

3. **Test model & Evaluate**
    - Run `./test.ipynb`.
    - Evaluate generated `choihong04_test_result.txt` file using `eval.pl`.

    ```sh
    perl eval.pl choihong04_test_result.txt test_label.txt
    ```

## Code files

1. **train.ipynb**
    ```sh
    Running `train.ipynb` trains the model using `model.py` and the training data, producing a `model.pkl` file.
    ```

2. **test.ipynb**
    ```sh
    Running `test.ipynb` applies preprocessing to the test data via model.py, then uses `model.pkl` to make classification predictions on the test set. The results are evaluated using two metrics—accuracy and F1-score—and the predicted labels are saved to `choihong_test_result.txt`. You can then run the `eval.pl` script to compute the evaluation metrics for those predictions.
    ```

