*	DETR (Detection Transformer): "어디에 **[상자]**가 있고, 그 상자가 **[무슨 물체]**인지 내놔라!"

![](https://github.com/gnosia93/vlm-distillation/blob/main/images/detr-arch.png)


*	SAM (Segment Anything Model): "내가 찍은 이 지점/상자 안의 **[물체 모양 겉테두리 마스크]**를 따서 내놔라!"
SAM 1/2 는 점·박스 같은 시각적 프롬프트, SAM 3 부터는 여기에 "yellow school bus" 같은 짧은 명사구(concept)가 추가되었다.
SAM 3 는 내부적으로 DETR 식 decoder(query 200개 → 상자 refine)를 쓰지만, "무슨 물체인가"를 고정 클래스 분류기로 판정하지 않고 텍스트 임베딩과의 dot product 로 계산한다.

![](https://github.com/gnosia93/vlm-distillation/blob/main/images/sam-arch.png)

