try: 
    import numpy as np
    from scipy.ndimage.interpolation import shift
except AssertionError as error:
    print(error) 

def image_shifting(df, target, vertical, horizontal): 
    dummy_coln = np.empty(df.shape)
    dummy_target = np.empty(target.shape)
    for i, (image,target) in enumerate(zip(df, target)):
        dummy_coln[i] = shift(image.reshape(28,28), [vertical, horizontal], cval= 0).reshape([-1])
        dummy_target[i] = target
    return {'features': np.r_[df, dummy_coln], 'target': np.r_[target, dummy_target]}
