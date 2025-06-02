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
└── train.ipynb             # train 코드드
```

## How to run

1. **필수 라이브러리 설치**
    ```sh
    pip install -r requirements.txt
    ```

2. **모델 학습**
    - `./train.ipynb` 파일을 실행하세요.

3. **예측 결과 평가**
    - 생성된 `team_test_result.txt` 파일을 `eval.pl`로 평가할 수 있습니다.
    ```sh
    perl eval.pl team_test_result.txt test_label.txt
    ```

## Reference

- 데이터 및 라벨 파일 구조는 `2501ml_data/label/` 폴더 참고
