# Matching Algorithm-FL

In a Federated Learning Concept we want to train models (one on each server) to detect natural disasters using images from Users.

In this implementation, before the FL procedure, we aim to create an Accurate Matching between Servers & Users to improve training.

Implementation:

1. Create Servers, Users and Critical Points Topology
2. Match Servers & Users using Approximate FedLearner
3. Match Servers & Users using Accurate FedLearner (takes as input the matching of the Approximate FedLearner)
4. Train the S models (one on each server) based on the Matching