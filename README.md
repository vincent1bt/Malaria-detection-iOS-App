# Malaria detection iOS App
This app classifies blood smear images as parasitized or uninfected using your device's back camera and TensorFlow Lite.

## Model
I used the Mobile Net v2 architecture and Keras to train the model.
The model has a **sensitivity of 95.8% and a specificity of 97%.**

You can see the code of this model in [this Jupyter notebook](https://nbviewer.jupyter.org/github/vincent1bt/Healthy-notebooks/blob/master/Malaria.ipynb).

## App
This app is written in Swift but it needs Objective C code to call TensorFlow Lite functions.



  