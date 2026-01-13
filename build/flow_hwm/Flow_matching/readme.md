Explaination:

Equations used:

1. x_t = t.x_target + (1-t).x_noise

where 
x_t interpolated noise

x_target: target distribution sample

x_noise: intial gaussian noise 

2. v_target = (x_target - x_noise)/(1-t)

where 1-t is used for flow conditional 

3. x_t = x_t + pred/step

p(x) : probability path/Ideal/target prob path

psi(x) : Flow paths

u_t(x) : Vector fields

Ideally learn from the flow fields/vector field

ODE only used for inference

v_t(x) : regression underlying vector field/neural network used for prediction


Reference github repo index: 
https://deepwiki.com/FedeAi/flow-matching/3.4-data-pipeline-and-conditioning


https://colab.research.google.com/drive/13lAveB3qwjkgyILWW-9qiOOSHG0U5_O6?usp=sharing

