ifndef BUILD_DIR
	BUILD_DIR:=.
endif

ifndef CONDA_ENV
	CONDA_ENV:=videoseal-deploy
endif

BUILD_DIR:=$(BUILD_DIR)/build
MODEL_DIR:=$(BUILD_DIR)/models/detector

GENS:=$(MODEL_DIR)/env.tar.gz $(MODEL_DIR)/config.pbtxt $(MODEL_DIR)/1/model.py $(MODEL_DIR)/1/checkpoints/.marker

.PHONY: all
all: ${GENS}

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)

$(MODEL_DIR)/env.tar.gz:
	rm -f $@
	mkdir -p $$(dirname $@)
	conda-pack -n $(CONDA_ENV) -o $@
	chmod +r $@

$(MODEL_DIR)/config.pbtxt: config.pbtxt
	mkdir -p $$(dirname $@)
	cp $< $@

$(MODEL_DIR)/1/model.py: model.py
	mkdir -p $$(dirname $@)
	cp $< $@

$(MODEL_DIR)/1/checkpoints/.marker:
	mkdir -p $$(dirname $@)
	cp /checkpoint/radkins/videoseal/* $$(dirname $@)
	touch $@