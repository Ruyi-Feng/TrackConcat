# TrackConcat
This project aims for broken trajectory concatenation through a joint model of NS_Transformer and Siamese-VGG16.

Under Review, Just For Display

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
### A Joint Spatiotemporal Prediction and Image Confirmation Model for Vehicle Trajectory Concatenation with Low Detection Rates
### under review
Ensuring the quality of trajectories is of utmost importance in traffic flow analysis. Traditional approaches rely on reconstructing nearly complete trajectories and subsequently denoising them. However, low detection rates often pose challenges and result in failed trajectory construction. To overcome this issue, this paper presents a trajectory concatenation method that combines NS Transformer prediction and Siamese-VGG16 similarity confirmation, specifically designed to address low detection rates. The employed transformer model is effective in mining internal associations and assessing the contributions of multiple traffic parameters. Furthermore, a lightweight image feature similarity verification step is integrated after trajectory prediction to find the most similar target to the image in the predicted spatiotemporal domain. Additionally, a lightweight image feature similarity verification step is integrated after trajectory prediction to identify the most similar targets within the predicted spatiotemporal domain. Experimental results demonstrate the efficacy of the proposed method, successfully connecting over 80\% of fragmented tracks and yielding significant maintenance of MOTA above 0.74 under low detection accuracy.