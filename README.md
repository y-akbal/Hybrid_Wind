# Attention Based Time Series Prediction

Put all your possibly correlated time series files into the same directory, with the header "ws". The time_series_slicer class will pick up the columns with "ws" header.

Then run:
pyhton hybrid_transformer.py --lr LR --batch_size BATCH_SIZE --epochs EPOCHS --decay_rate DECAY_RATE

Some notes: 1) Decay rate here refers to decay rate of the learning rate,
2) A single GPU is mostly enough for any kind of purposses.

