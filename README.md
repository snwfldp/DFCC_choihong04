# DFCC (DeepFake speech Classification Challenge)
인천대학교 기계학습 기말 프로젝트 베이스라인 코드입니다.  
이 저장소는 딥페이크 음성 탐지를 위한 베이스라인 코드와 데이터 예시를 제공합니다.

## 폴더 구조

```
.
├── 2501ml_data             # 2501ml_data.zip 압축 해제
│   ├── label
│   ├── test
│   └── train
├── baseline_DNN            # DNN 베이스라인 코드
├── baseline_GMM            # GMM 베이스라인 코드
├── baseline_SVM            # SVM 베이스라인 코드
├── eval.pl                 # 평가 스크립트
├── librosa_example.ipynb   # librosa 특징 추출 예제
└── requirements.txt        # 필요 라이브러리 목록
```

## 베이스라인 코드 실행 방법

1. **필수 라이브러리 설치**
    ```sh
    pip install -r requirements.txt
    ```

2. **특징 추출 및 모델 학습/평가**
    - 각 베이스라인 폴더(`baseline_DNN`, `baseline_GMM`, `baseline_SVM`)의 `.ipynb` 파일을 실행하세요.
    - 데이터 경로는 이미 코드에 지정되어 있습니다.

3. **예측 결과 평가**
    - 각 베이스라인에서 생성된 `team_test_result.txt` 파일을 `eval.pl`로 평가할 수 있습니다.
    ```sh
    perl eval.pl team_test_result.txt test_label.txt
    ```

## 참고

- [librosa_example.ipynb](librosa_example.ipynb): 오디오 특징 추출 예제
- 데이터 및 라벨 파일 구조는 `2501ml_data/label/` 폴더 참고
