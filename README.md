# Bayesian Neural Networks with Uncertainty measure

This project implements a Bayesian approach to neural network classification on the MNIST dataset. By incorporating weight uncertainty into neural networks, we enable the model to express uncertainty in its predictions, providing a more nuanced understanding of its decision-making process.

## Overview of the Research

Traditional neural networks use fixed values for weights, leading to models that are often overconfident in their predictions. This research explores a Bayesian treatment of neural networks, allowing for a distribution over weights, which introduces the notion of uncertainty into the model's predictions. This is achieved through a novel variational inference method known as "Bayes by Backprop."

## Project Structure

- `BayesianNeuralNetwork`: The core class that defines the Bayesian neural network architecture.
- `nonlinearity`, `log_gaussian`, `log_gaussian_logsigma`, `get_random`, `log_categ`: Utility functions for the model's operations.
- Main training loop: Code block that prepares the MNIST dataset, initializes the Bayesian Neural Network, and conducts the training process.
- Visualization: Scripts to generate various plots for analyzing the model's performance and behavior.

## Visualization Results

![Prediction Confidence Histogram](./Figure_1.png)
*Figure 1: Histogram of prediction confidence scores, showcasing the model's certainty in its predictions.*

![Weight Distributions](./Figure_2.png)
*Figure 1: Weight distributions across different layers of the Bayesian Neural Network.*

![Feature Importance](./Figure_5.png)
*Figure 3: Histogram of prediction confidence scores, showcasing the model's certainty in its predictions.*

![Class Probabilities and Uncertainty Intervals](./Figure_4.png)
*Figure 4: Probabilities and uncertainty intervals for each class, providing insight into the model's certainty across different classes.*

## Requirements

- Python 3.x
- PyTorch
- NumPy
- Matplotlib
- scikit-learn

## Usage

To run the project, execute the main script:
`python bayesian_nn_script.py`


This will start the training process and display the visualizations upon completion.

## Citation

https://arxiv.org/abs/1505.05424
Blundell, C., Cornebise, J., Kavukcuoglu, K., & Wierstra, D. (2015). Weight Uncertainty in Neural Networks. arXiv preprint arXiv:1505.05424.

BibTeX entry:
```bibtex
@article{blundell2015weight,
  title={Weight Uncertainty in Neural Networks},
  author={Blundell, Charles and Cornebise, Julien and Kavukcuoglu, Koray and Wierstra, Daan},
  year={2015},
  eprint={1505.05424},
  archivePrefix={arXiv},
  primaryClass={stat.ML}
}```

```bibtex
@article{simonyan2014very,
  title={Very deep convolutional networks for large-scale image recognition},
  author={Simonyan, Karen and Zisserman, Andrew},
  journal={arXiv preprint arXiv:1409.1556},
  year={2014}
}```

```bibtex
@article{szegedy2016rethinking,
  title={Rethinking the Inception Architecture for Computer Vision},
  author={Szegedy, Christian and Vanhoucke, Vincent and Ioffe, Sergey and Shlens, Jon and Wojna, Zbigniew},
  journal={arXiv preprint arXiv:1512.00567},
  year={2016}
}```

```bibtex
@article{dosovitskiy2020image,
  title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and others},
  journal={arXiv preprint arXiv:2010.11929},
  year={2020}
}```

