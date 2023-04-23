# TrackConcat
This project aims for broken trajectory concatenation through a joint model of NS_Transformer and Siamese-VGG16.



## Framework
1. time-space（NS-Transformer）
- input： broken trajectories，call predict
- output：possible range
2. image feature (Siamese-VGG16)
- input：interference candidate image in possible range
- output：similarity
3. fusion
- input：1. All candidate distance  2. Similarity
- output：target


## Overall guaidance
### Input
- raw trajectory csv file path
- cooresponding image screenshot named by car_id
- some setting parameters
```
# args, params, dir, flnm
```
### Output
- refreashed csv with trajetcories matched and gap completed

## Paper
### A Joint Spatiotemporal Prediction and Image Confirmation Model for Vehicle Trajectory Concatenation  with Low Detection Rates
Abstract: This paper proposes a concatenation method adapted to low detection rate. In the proposed method, a location fused Transformer are arranged for predicting future possible range with consideration of dynamic of traffic flow. All candidates within the predicted range are assigned an initial confidence. Then, convolutional based algorithm generates image similarity between candidates to adjusting the confidence, realizing concatenation with a fusion of time-spatial and image features. The structure of the remaining of this paper follows: Section 2 introduces detailed algorithm modules of range prediction and similarity adjustment. Section 3 presents the scenarios and basic data for the experimental design. Section 4 analyzes experimental results separately of modules and equipped a sensitivity analysis. Section 5 discusses and summarizes the contribution and application prospects.
