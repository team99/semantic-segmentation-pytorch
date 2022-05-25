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

train_watermark_cpu:
	python train.py --cfg config/$(CONFIG)

# Run training with GPU
train_watermark_gpu:
	python train.py --cfg config/$(CONFIG) --gpu $(GPU)

# Only run this after you run training model script for watermark
# Check whether imgs, cfg, DIR folder exists before running this
# Just update the file directory path to use different image, config, or checkpoint
test_watermark_cpu:
	python test.py \
		--imgs watermark_data/vietnam/images/validation \
		--cfg config/$(CONFIG) \
		DIR ckpt/$(CKPT) \
		TEST.result watermark_data/vietnam/output/validation \
		TEST.checkpoint epoch_1.pth

# Run test with GPU
test_watermark_gpu:
	python test.py \
		--imgs watermark_data/vietnam/images/validation \
		--cfg config/$(CONFIG) \
		--gpu $(GPU) \
		DIR ckpt/$(CKPT) \
		TEST.result watermark_data/vietnam/output/validation \
		TEST.checkpoint epoch_1.pth