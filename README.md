# Gradient-based Search for Activation Functions
### Idea: We apply Gradient based one-shot methods to search for novel activation functions and evaluate the results with respect to their performance and transferability
### Sharat Patil, Jake Robertson, Lukas Strack                                                              
#### Supervisor: Mahmoud Safari


### Search:
RNNResNet8_restricted.py  
search for activations functions on ResNet8 with restricted search space

#### Settings:
config.optimizer = 'darts'  # 'gdas', 'drnas' 

search_space = ActivationFuncResNet8SearchSpace("small") # simple activation cell
search_space = ActivationFuncResNet8SearchSpace("huge") # complex activation cell

#### Output:
Files in : run/cifar10/Optimizer_name/nth_run/  
errors.json -> Accuracy and loss values  
log.log -> training log. Will contain final discretized cell  

analyse_search_results.ipynb -> functionality to visualise the alphas

### Evaluation:
run_eval_r.py --network ResNet8 --ac_func 0 --seed 0  

Evaluates the activation function on Cifar10

Networks:
ResNet8  
ResNet20  

Activation functions:  
0: Darts_simple_r  
1: Darts_complex_r  
2: ReLU  
3: SiLU  
4: DrNas_simple_r  
5: DrNas_complex_r  
6: Gdas_simple_r
#### Output:
Files in /eval/eval_network_acfunc_seed/  
errors.json -> Accuracy and loss values  
model.pth -> state_dict of final trained model  

### Results:
[Poster](Deep_Learning_Lab_Poster.pdf)

### References
[1] Ramachandran, Prajit, Barret Zoph, and Quoc V. Le. "Searching for activation functions." (2017).  
[2] Liu, Hanxiao, Karen Simonyan, and Yiming Yang. "Darts: Differentiable architecture search." (2018).  
[3] Dong, X., & Yang, Y. (2019). Searching for A Robust Neural Architecture in Four GPU Hours (Version 2).   
[4] Chen, X., Wang, R., Cheng, M., Tang, X., & Hsieh, C.-J. (2020). DrNAS: Dirichlet Neural Architecture Search (Version 4).  
[5] Ruchte, Michael, et al. "NASLib: a modular and flexible neural architecture search library." (2020).




    