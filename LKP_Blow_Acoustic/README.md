# Simulation Instructions

## Training

Before training the model, specify the type of model architecture in the `train.py` file as follows:
- MODEL_TYPE = "AUTOENCODER" //for the autoencoder archiecture
- MODEL_TYPE = "VAE" //for the variational autoencoder archiecture
- MODEL_TYPE = "TRANSFORMER_AUTOENCODER" //for the transformer autoencoder archiecture

Then, run the following:
- `python train.py`

## Evaluation

To evaluate its accuracy and runtime performance, first specify the following parameters in the `main.py` file as follows:
-  MODEL_TYPE = # Specify the same model type you specified during training.
-  THRESHOLD_MODE = # Choose either "GLOBAL" or "PER-USER".

Then, run the following:
- `python main.py`

At the end of the simulation, you can see the accuracy reports, such as "Accuracy", "FAR", "FRR" and "EER". You ca also see the average evaluation time per each evaluation sample. 
