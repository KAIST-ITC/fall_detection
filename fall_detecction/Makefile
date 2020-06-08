help:
	@echo 'data_ready      - Pad or cut the extra frames to make the size of'
	@echo '                  the radar data the same among the original data'
	@echo '                  and create data_ready dir to store the new data'
	@echo ''
	@echo 'train           - With /configs/config.json file'
	@echo '                  train and test the network'
	@echo ''
	@echo 'clean           - undo make data'

data_ready:
	python data_wranggle.py

.PHONY: train
train:
	python train.py -c "./configs/config.json"

.PHONY: clean
clean:
	if [ -d data_ready/ ]; then rm -r data_ready/; fi
	if [ -d __pycache__/ ]; then rm -r __pycache__/; fi
