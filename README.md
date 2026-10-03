
# Rotation-aware Fashion Image Retrieval 실험 결과

## 1. 실험 목적

회전된 fashion query image에 대해 다음 세 가지 검색 방식을 비교했다.

1. **Baseline**
   - 회전된 query를 그대로 CLIP embedding
   - cosine similarity로 관련 상품 검색

2. **Rotation Classifier**
   - ResNet18로 `0° / 90° / 180° / 270°` 방향 예측
   - 예측된 각도의 반대 방향으로 image correction
   - CLIP embedding 후 cosine similarity 검색

3. **Max4**
   - query를 `0° / 90° / 180° / 270°` 네 방향으로 각각 회전
   - 각 방향에서 CLIP embedding 계산
   - 각 gallery image에 대해 4개 cosine similarity 중 최대값 사용

---

## 2. Dataset 및 평가 설정

- Dataset: `ashraq/fashion-product-images-small`
- 전체 데이터: **44,072**
- Apparel subset: **21,361**
- Rotation model train: **8,000**
- Rotation model validation: **1,000**
- Retrieval gallery: **10,000**
- Retrieval query: **1,000**
- Query와 Gallery의 product ID는 서로 겹치지 않음
- 따라서 동일 이미지 검색이 아닌 **관련 상품 검색** 실험

### Relevance 정의

nDCG 계산에는 graded relevance를 사용했다.

| Gain | 조건 |
|---:|---|
| 3 | same `articleType` + `gender` + `baseColour` |
| 2 | same `articleType` + `gender` |
| 1 | same `subCategory` + `gender` |
| 0 | 그 외 |

`Precision / Recall / mAP`에서는 다음을 relevant item으로 정의했다.

```text
gain >= 2
= same articleType + same gender
```

---

## 3. Rotation Classifier 성능

| Epoch | Train Accuracy | Validation Accuracy |
|---:|---:|---:|
| 1 | 93.33% | 99.20% |
| 2 | 99.38% | 99.00% |
| 3 | **99.68%** | **99.50%** |

실제 retrieval query에서의 rotation classification accuracy도 매우 높았다.

| Query Rotation | Rotation Accuracy |
|---:|---:|
| 0° | 99.80% |
| 90° | 99.70% |
| 180° | **99.90%** |
| 270° | 99.30% |

4-way orientation classification 자체는 현재 데이터에서 거의 포화 수준의 성능을 보였다.

---

# 4. Retrieval 결과

## 4.1 0° — 정상 방향

| Method | nDCG@10 | P@10 | R@10 | mAP | Latency |
|---|---:|---:|---:|---:|---:|
| Baseline | **0.5559** | **0.7853** | 0.0389 | **0.5645** | **1.70 ms** |
| Rotation Classifier | **0.5559** | **0.7853** | 0.0389 | **0.5645** | 2.20 ms |
| Max4 | 0.5555 | 0.7842 | 0.0389 | 0.5499 | 7.23 ms |

정상 이미지에서는 Rotation Classifier를 추가해도 검색 성능이 그대로 유지됐다.

반면 Max4는 nDCG 차이는 거의 없지만 mAP가 `0.5645 → 0.5499`로 소폭 하락했다.

즉 정상 이미지에서 여러 orientation의 similarity를 무조건 max하는 방식은 일부 irrelevant candidate의 score까지 높일 가능성이 있다.

---

## 4.2 90° Rotation

| Method | nDCG@10 | P@10 | R@10 | mAP | Latency |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.3045 | 0.4054 | 0.0183 | 0.2852 | **1.95 ms** |
| Rotation Classifier | **0.5527** | **0.7795** | 0.0384 | **0.5625** | 2.18 ms |
| Max4 | 0.5522 | 0.7777 | 0.0384 | 0.5444 | 7.01 ms |

90° 회전에서는 baseline 검색 성능이 크게 감소했다.

Rotation correction을 적용하면 nDCG@10이

```text
0.3045 → 0.5527
```

로 거의 정상 이미지 수준까지 복구됐다.

---

## 4.3 180° Rotation

| Method | nDCG@10 | P@10 | R@10 | mAP | Latency |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.2071 | 0.2700 | 0.0159 | 0.2245 | **1.85 ms** |
| Rotation Classifier | **0.5559** | **0.7853** | 0.0389 | **0.5644** | 2.18 ms |
| Max4 | 0.5555 | 0.7842 | 0.0389 | 0.5499 | 5.10 ms |

180°가 baseline에서 가장 큰 성능 저하를 보였다.

```text
nDCG@10
0°   : 0.5559
180° : 0.2071
```

약 **63% 감소**한다.

반면 Rotation Classifier 적용 후에는:

```text
0°   : 0.5559
180° : 0.5559
```

로 사실상 완전히 복구됐다.

---

## 4.4 270° Rotation

| Method | nDCG@10 | P@10 | R@10 | mAP | Latency |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.3041 | 0.4069 | 0.0176 | 0.2849 | **2.05 ms** |
| Rotation Classifier | **0.5522** | **0.7786** | 0.0379 | **0.5616** | 2.29 ms |
| Max4 | **0.5522** | 0.7777 | **0.0384** | 0.5444 | 5.46 ms |

270°에서도 90°와 거의 동일한 패턴을 보였다.

---

# 5. 핵심 결과: 회전 이미지 평균

`90° / 180° / 270°`의 평균 결과는 다음과 같다.

| Method | nDCG@10 | P@10 | R@10 | mAP | Latency |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.2719 | 0.3608 | 0.0173 | 0.2648 | **1.95 ms** |
| **Rotation Classifier** | **0.5536** | **0.7811** | 0.0384 | **0.5628** | **2.22 ms** |
| Max4 | 0.5533 | 0.7799 | **0.0385** | 0.5462 | 5.86 ms |

### Baseline → Rotation Classifier 개선

```text
nDCG@10 : 0.2719 → 0.5536   (+103.6%)
P@10    : 0.3608 → 0.7811   (+116.5%)
R@10    : 0.0173 → 0.0384   (+122.0%)
mAP     : 0.2648 → 0.5628   (+112.5%)
```

관련 상품 검색에서 orientation correction의 효과가 매우 크다.

---

# 6. Rotation Classifier vs Max4

검색 품질만 비교해도 Rotation Classifier가 Max4보다 열세가 아니다.

| Metric | Rotation Classifier | Max4 |
|---|---:|---:|
| nDCG@10 | **0.5536** | 0.5533 |
| P@10 | **0.7811** | 0.7799 |
| R@10 | 0.0384 | **0.0385** |
| mAP | **0.5628** | 0.5462 |
| Latency | **2.22 ms** | 5.86 ms |

nDCG와 P@10은 사실상 동등하지만, mAP에서는 Rotation Classifier가 더 높다.

```text
mAP
Rotation Classifier = 0.5628
Max4                = 0.5462
```

특히 latency 차이가 크다.

```text
Rotation Classifier : 2.22 ms/query
Max4                : 5.86 ms/query
```

Max4가 약 **2.64배 느리다.**

Rotation Classifier는 baseline 대비 latency가:

```text
1.95 → 2.22 ms
```

약 **14% 증가**하는 수준이다.

---

# 7. Max4가 더 좋지 않았던 이유

Max4는 각 gallery candidate에 대해

```text
max(
    sim(query_0°, gallery),
    sim(query_90°, gallery),
    sim(query_180°, gallery),
    sim(query_270°, gallery)
)
```

를 사용한다.

이 방식은 올바른 orientation을 포함할 수 있다는 장점이 있지만, 동시에 **잘못된 orientation에서 우연히 높은 similarity를 가지는 irrelevant item의 score도 증가시킬 수 있다.**

실제로 정상 방향에서도:

```text
Baseline mAP = 0.5645
Max4 mAP     = 0.5499
```

로 감소했다.

따라서 모든 orientation score 중 maximum을 선택하는 방식은 검색 recall 측면에서는 안정적일 수 있지만 ranking precision을 악화시킬 가능성이 있다.

---

# 8. Recall@10이 낮은 이유

Rotation Classifier의 `P@10 ≈ 0.78`인데 `R@10 ≈ 0.038`인 것은 모순이 아니다.

현재 relevant 정의가:

```text
same articleType + same gender
```

이기 때문에 gallery 10,000개 안에 관련 상품이 상당히 많이 존재한다.

예를 들어 relevant item이 200개라면 top-10에서 8개를 맞혀도:

```text
Precision@10 = 8 / 10
             = 0.80

Recall@10 = 8 / 200
          = 0.04
```

가 된다.

따라서 현재 실험에서는 `nDCG@10`, `P@10`, `mAP`가 top-ranking 품질을 판단하는 데 더 직접적인 지표다.

---

# 9. 결론

현재 실험에서는 **Rotation Classifier를 이용해 이미지를 canonical orientation으로 복원한 후 한 번 검색하는 방식이 가장 합리적이다.**

추천 pipeline은 다음과 같다.

```text
Input image
     ↓
Rotation classifier
0 / 90 / 180 / 270
     ↓
Inverse rotation
     ↓
CLIP / VLM embedding
     ↓
ANN search
     ↓
Related fashion products
```

실험 결과 Rotation Classifier는 회전된 이미지의 nDCG@10을:

```text
0.2719 → 0.5536
```

으로 복구했으며, 정상 이미지 검색 성능 `0.5559`와 사실상 동일한 수준이다.

동시에 Max4보다:

- nDCG@10: 사실상 동일
- P@10: 소폭 우수
- mAP: **우수**
- latency: **약 62% 낮음**

을 보였다.

따라서 현재 조건에서는 **4방향으로 각각 embedding하고 max cosine을 계산할 이유가 크지 않다.**

## 최종 판단

```text
Baseline
→ 회전에 매우 취약

Rotation Classifier + correction
→ 검색 성능 거의 완전 복구
→ 낮은 추가 latency
→ 현재 실험의 가장 좋은 선택

4-way Max Cosine
→ 검색 품질 향상은 Rotation Classifier와 거의 동일
→ mAP는 오히려 낮음
→ 약 2.6× 높은 latency
```

**현재 결과만 기준으로 하면 `Rotation Classifier → inverse rotation → single embedding search`를 채택하는 것이 가장 적절하다.**

---

## 10. 실험의 한계

현재 결론은 `0° / 90° / 180° / 270°` discrete rotation 문제에 대해서는 강하지만, 다음 케이스까지 일반화됐다고 보기는 어렵다.

- 20°, 45°, 70° 같은 arbitrary rotation
- perspective distortion
- 실제 촬영 시 camera tilt
- partial occlusion
- crop된 상품
- background가 복잡한 consumer image

또한 rotation model도 정확히 같은 네 방향을 classification하도록 학습했기 때문에 현재 문제는 상당히 잘 정의된 closed-set orientation classification이다.

따라서 다음 단계에서는 **arbitrary angle까지 포함한 continuous rotation correction**을 실험하면 rotation model이 실제 production 환경에서도 필요한지 더 강하게 검증할 수 있다.
