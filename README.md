# Matching Algorithm-FL

[![Python Version](https://img.shields.io/badge/python-3.6%2B-blue.svg)](https://www.python.org/downloads/release)
[![TensorFlow Version](https://img.shields.io/badge/tensorflow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Keras Version](https://img.shields.io/badge/keras-2.x-red.svg)](https://keras.io/)

In a Federated Learning Concept we want to train models (one on each server) to detect natural disasters using images from Users.

In this implementation, before the FL procedure, we aim to create different Matchings between Servers & Users to improve training.

Matchings: Random, Game Theory (Approximate FedLearner --> Accurate FedLearner), Reinforced Learning (Using different mechanisms)

Implementation:

1. Create Servers, Users and Critical Points Topology
2. Calculate the parameters of each User and Server for the problem
3. Match Servers & Users using a Matching
4. Train the S models (one on each server) based on the Matching
5. Repeat from step 3 for other Matchings and log the results into files for comparison