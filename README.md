# ✨ YOLO 11 & YOLO 12: 전문 분석 README 목차 ✨

최신 객체 탐지 모델 **YOLO 11**과 **YOLO 12**의 아키텍처 및 기술 세부 사항을 한눈에 파악할 수 있는, 구조적으로 미려하며 연구 환경에 적합한 README 목차 예시입니다. 각 항목을 선택하면 해당 주제의 상세한 마크다운 문서로 바로 이동할 수 있습니다.

---

## 📚 목차

### 🦾 YOLO 11
- [개요 및 주요 목표](./YOLO11_Overview.md)
- [네트워크 아키텍처 상세 분석](./YOLO11_Architecture.md)
    - Backbone 구조
    - Neck/Feature Aggregation
    - Detection Head 및 Output Layer
- [학습 방법 및 Loss Function](./YOLO11_Training_Strategy.md)
    - [Data Augmentation 기법](./YOLO11_Training_DataAugmentation.md)
    - [Anchor Box/Free Anchor 전략](./YOLO11_Training_Anchor.md)
    - [정규화/Regularization 기법](./YOLO11_Training_Regularization.md)
- [성능 벤치마크 및 실험 결과](./YOLO11_Benchmark.md)
    - [COCO 및 Open Dataset 성능](./YOLO11_Benchmark_Coco.md)
    - [속도-정확도 트레이드오프](./YOLO11_Benchmark_Tradeoff.md)
    - [Ablation Study](./YOLO11_Benchmark_Ablation.md)
- [적용 사례 및 한계점 분석](./YOLO11_Applications.md)
    - [실환경 배포 사례](./YOLO11_Applications_Deployment.md)
    - [실패 원인 & 개선점](./YOLO11_Applications_Limitations.md)

---

### 🚀 YOLO 12
- [개요 및 핵심 기술 발전](./YOLO12_Overview.md)
- [혁신적인 아키텍처 설계](./YOLO12_Architecture.md)
    - [Transformer/Attention 기반 모듈](./YOLO12_Architecture_Attention.md)
    - [Multi-Scale Feature Fusion](./YOLO12_Architecture_MultiScale.md)
    - [Head 구조 개량점](./YOLO12_Architecture_AdvancedHead.md)
- [학습 및 최적화 전략](./YOLO12_Training_Strategy.md)
    - [Self-supervision/Pre-training](./YOLO12_Training_SelfSupervised.md)
    - [Hyperparameter Tuning 사례](./YOLO12_Training_Hyperparameters.md)
    - [효과적인 학습 스케줄링](./YOLO12_Training_Scheduling.md)
- [정량적 성능 평가](./YOLO12_Benchmark.md)
    - [다중 데이터셋 비교 분석](./YOLO12_Benchmark_Dataset.md)
    - [추론 속도 및 메모리 사용](./YOLO12_Benchmark_Inference.md)
    - [Ablation & ablation study](./YOLO12_Benchmark_Ablation.md)
- [실제 응용 및 한계점](./YOLO12_Applications.md)
    - [산업별 적용 사례](./YOLO12_Applications_Industry.md)
    - [미래 과제 및 오픈 이슈](./YOLO12_Applications_Future.md)

---

