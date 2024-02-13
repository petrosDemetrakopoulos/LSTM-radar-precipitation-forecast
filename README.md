## Short-term precipitation forecasting from weather radar data using Convolutional LSTM neural networks
This is the repository with the code accompanying this [Medium article](https://medium.com/@petrosdemetrakopoulos/short-term-precipitation-forecasting-using-convolutional-lstm-neural-networks-f347db1b5f1d).The relevant paper is also available in [ArXiv](https://arxiv.org/abs/2312.01197)
## Data
The raw data used for training and validation can be found [in this link](https://drive.google.com/drive/folders/17MCuvQmcydUSNUFwf_rZLtFt2K0kddlV?usp=sharing) (too large to host on GitHub). A trained version of the model can also be found in the same link for inference purposes.
## The model
The model was developed using Tensorflow and Keras
![Model](./model.png)
## Results
![Ground Truth](./ground_truth.gif)

![Predicted frames](./predicted.gif)

## Cite
```
@misc{demetrakopoulos2023shortterm,
      title={Short-term Precipitation Forecasting in The Netherlands: An Application of Convolutional LSTM neural networks to weather radar data}, 
      author={Petros Demetrakopoulos},
      year={2023},
      eprint={2312.01197},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
