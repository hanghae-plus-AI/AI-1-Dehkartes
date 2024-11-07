# Runtime (x: step, y: Runtime)
https://api.wandb.ai/links/dehkartes-fractarium/nr0ea49p
 - 랭크가 낮을수록 속도 빠름
# train/loss (x: step, y: train/loss)
https://api.wandb.ai/links/dehkartes-fractarium/po5jg1t6
- 470 step 부터 랭크 별 loss 격차가 벌어지기 시작했다
- 랭크가 높을수록 스텝 당 로스 감소 큼

# 메모리 점유율 (x: step, y: memory)
https://api.wandb.ai/links/dehkartes-fractarium/66xjtt6h
- 랭크가 낮을수록 점유율 낮음

# 결론
- 속도와 메모리 점유율 측면 에서는 낮은 랭크가 유리
- step 당 loss 감소 측면 에서는 높은 랭크가 유리

- 복잡한 문제를 해결 하기 위해서는 높은 랭크로 모델 정확도를 확보
- 시간이나 자원이 부족할때는 낮은 랭크로 빠르게 모델 학습