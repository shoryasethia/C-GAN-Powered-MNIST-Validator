from backtester import *

# specify you model's path below
model_path = "mnist-cnn.h5"

# loading model
model = models.load_model(model_path)


# model : takes your model as input
# plot = 1 : plot some predictions and ground truth
# plot = 0 : figures won't be plotted
# num_test_images : number of images on which you want to test your model

model_check(model = model, plot = 1, num_test_images=1000)