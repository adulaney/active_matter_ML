# Machine Learning for Phase Behavior in Active Matter
This package is used to predict the phase of individual particles in suspensions of active colloids.

## Dependencies
These scipts have been tested using Python 3.7.8, with the following packages (and their dependencies):
- pandas==1.1.2
- scipy==1.5.2
- scikit-learn==0.23.2
- tensorflow-gpu==2.2.0
- glob2==0.7
- pytorch==1.3.1
- xgboost==1.1.0
- hyperopt==0.2.4

## Packages for simulation inputs
- HOOMD==2.3.4
- gsd==2.2.0

Additionally, CUDA 10.2 and cuDNN 7 have been used.

## Reference
If you make use of these models or methods in your research, please cite the following in your manuscript:

	@article{
    dulaney2020machine,
    title="{Machine Learning for Phase Behavior in Active Matter}",
    author{Dulaney, Austin R. and Brady, John F.},
    journal={arXiv preprint arXiv:2011.09458},
    year={2020}
  }


## License
MIT
