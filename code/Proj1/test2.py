
import numpy as np
from operator import itemgetter
    
    
a = np.array([1,2,3])
b = np.array([31,7,9])


c =np.array([1,2,3])
cleaned_data = zip(c, abs(a-b))
cleaned_data.sort(key=itemgetter(1))
print cleaned_data[0:2]
        