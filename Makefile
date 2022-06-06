PYTHON = 3.7.13
PIP = $(shell which pip3)
WORKDIR = $(shell pwd)
GPU = 1
CKPT = vietnam-resnet50dilated-ppm_deepsup
CONFIG = $(CKPT).yaml


install:
	$(PIP) install --no-cache-dir -r requirements.txt

demo_test:
	source ./demo_test.sh

##### TRAINING #####
train_watermark_cpu:
	python train.py --cfg config/$(CONFIG)

# Run with GPU
train_watermark_gpu:
	python train.py --cfg config/$(CONFIG) --gpu $(GPU)

##### EVALUATION #####
eval_watermark_cpu:
	python eval_multipro.py --cfg config/$(CONFIG)

# Run with GPU
eval_watermark_gpu:
	python eval_multipro.py --cfg config/$(CONFIG) --gpu $(GPU)

##### TESTING #####
# Only run this after you run training model script for watermark
test_watermark_cpu:
	python test.py \
		--imgs watermark_data/vietnam/images/validation \
		--cfg config/$(CONFIG)

# Run with GPU
test_watermark_gpu:
	python test.py \
		--imgs watermark_data/vietnam/images/validation \
		--cfg config/$(CONFIG) \
		--gpu $(GPU)