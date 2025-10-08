# Newton Methods for Smoothing in State-Space Models

A toolbox for second-order batch and recursive smoothing methods.

## Installation
 
 Create a conda environment
    
    conda create -n NAME python=3.9
    
 Then head to the cloned repository and execute
 
    pip install -e .
    
 ## Examples
 
 An example of smoothing using a recursive Newton's method
 
    python examples/recursive/recursive_tr_newton.py
    
## Cite
```bib
@inproceedings{yaghoobi2023recursive,
  title={A Recursive Newton Method for Smoothing in Nonlinear State Space Models}, 
  author={Yaghoobi, Fatemeh and Abdulsamad, Hany and Särkkä, Simo},
  booktitle={2023 31st European Signal Processing Conference (EUSIPCO)}, 
  year={2023},
}
