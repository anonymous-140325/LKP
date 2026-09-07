# Simulation Instructions

## Training

Before training the model, specify the model architecture in the `train.py` file by setting `MODEL_TYPE` to one of the following:
- MODEL_TYPE = "AUTOENCODER" //for the autoencoder archiecture
- MODEL_TYPE = "VAE" //for the variational autoencoder archiecture
- MODEL_TYPE = "TRANSFORMER_AUTOENCODER" //for the transformer autoencoder archiecture

Then, run the following command to train the model using the dataset:
- `python train.py`

## Evaluation

To evaluate the model's accuracy and runtime performance, first specify the following parameters in the `main.py` file:
-  MODEL_TYPE = # Specify the same model type used during training.
-  THRESHOLD_MODE = # Choose either "GLOBAL" or "PER-USER".

Then, run the following command to perform the evaluation:
- `python main.py`

At the end of the simulation, you can see the accuracy reports, such as "Accuracy", "FAR", "FRR" and "EER". The simulation will also report the average evaluation time per sample. 
