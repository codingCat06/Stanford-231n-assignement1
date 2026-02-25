# CS231n Assignment 1 정리

현재 진행 상황:
- Q1 완료
- Q2 ~ Q5 진행 예정


# Q1. KNN Distance Computation 구현

## 1. dtype의 중요성 (이미지 데이터와 정수 타입 문제)
이미지 데이터는 일반적으로 uint8 (0~255 범위의 정수형)으로 저장되어 있다. uint8은 음수를 표현하지 못하기 때문에 뺄셈 결과가 음수가 되어야 하는 상황에서도 값이 wrap-around 되어 의도하지 않은 큰 양수로 변환되며 이는 거리 계산에서 치명적인 오류를 발생시킨다.


```python
X[i] - X_train[j]
```



### 해결 방법

수치 연산 전 반드시 float 타입으로 변환

```python
X = X.astype(np.float64)
X_train = X_train.astype(np.float64)
```

---

## 2. Broadcasting과 벡터화 구현

### 2.1 Two Loops (Baseline)

```python
for i in range(num_test):
    for j in range(num_train):
        dists[i, j] = np.sqrt(np.sum((X[i] - X_train[j]) ** 2))
```

### 2.2 No Loop Version 1 (3D diff 생성)

```python
diff = X[:, None, :] - X_train[None, :, :]
dists = np.sqrt(np.sum(diff ** 2, axis=2))
```

특징:
- Python loop 제거
- 빠름
- 하지만 (num_test, num_train, dim) 크기의
  거대한 3D 배열 생성 → 메모리 비효율


### 2.3 No Loop Version 2 (행렬 기반 최적화)

L2 distance 항등식 사용:

||x - y||^2 = ||x||^2 + ||y||^2 - 2 x·y

```python
test_sq = np.sum(X ** 2, axis=1)
train_sq = np.sum(X_train ** 2, axis=1)
cross = X @ X_train.T

dists = np.sqrt(
    np.maximum(test_sq[:, None] + train_sq[None, :] - 2 * cross, 0.0)
)
```

장점:
- 3D 배열 생성 안 함
- 메모리 효율적
- 가장 빠름
- 선형대수적 사고 적용


## 3. 성능 비교

| 방법 | 실행 시간 |
|------|------------|
| Two Loops | 38s |
| One Loop | 37s |
| No Loop | 1s |

→ 약 38배 이상의 차이 발생


## Q1을 통해 얻은 핵심 통찰

- dtype은 단순 형식 문제가 아니라 정확도 문제
- Broadcasting은 성능 최적화의 핵심
- 벡터화는 코드 스타일이 아니라 알고리즘적 사고

---
