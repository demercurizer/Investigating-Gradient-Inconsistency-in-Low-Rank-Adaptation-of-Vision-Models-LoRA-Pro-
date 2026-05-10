.PHONY: install train eval zero-shot count-params

install:
	pip install -r requirements.txt

train:
	python scripts/train.py --config configs/lorapro_eurosat_r8.yaml

eval:
	python scripts/eval.py --config configs/lorapro_eurosat_r8.yaml

zero-shot:
	python scripts/zero_shot.py --config configs/zero_shot_eurosat.yaml

count-params:
	python scripts/count_params.py --config configs/lorapro_eurosat_r8.yaml
