`make data_ready` to cut or pad radar data in ./data so that the number of frames is all 25. This command will generate new data and store them in ./data_ready

`make train` to train the model specified in train.py. Make changes in ./configs/config.json to change configurations. Training models are implemented in model.py, and what model to train is specified in train.py.
