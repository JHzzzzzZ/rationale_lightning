import numpy as np


<<<<<<< HEAD
decay = 1e-3
=======
decay = 1e-2
>>>>>>> 6a23cd0e257b930f0e51b52fc0b1340886592554
init_penalty=5e-3
aim = 0.5
if __name__ == '__main__':
    # np.exp(decay * step)*init_penalty == 1
    assert init_penalty > 0
    assert decay > 0
    assert aim > 0

    ans = np.ceil(np.log(aim/init_penalty)/decay)
    print(f'the total step with the decay {decay}, initial penalty {init_penalty} and the aim value {aim} is {ans}')