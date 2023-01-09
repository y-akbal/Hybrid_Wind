# Hybrid Transformer Model for Time Series Prediction

Put all your possibly correlated time series files into the same directory, with the headers "ws". The time_series_slicer class will pick up the columns with "ws" header. Slice them up, into sliding windows with window size  512. 

Then run:
pyhton hybrid_transformer.py --lr LR --batch_size BATCH_SIZE --epochs EPOCHS --decay_rate DECAY_RATE

Here we picked: lr = 0.001, batch_size = 64, epochs = 15, decay_rate = -0.001

Some notes: 1) Decay rate here refers to decay rate of the learning rate,
2) A single GPU is mostly enough for any kind of purposses.

