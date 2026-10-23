#%%
import sys
from pathlib import Path

import einops
import numpy as np
import torch as t
from torch import Tensor

# Make sure exercises are in the path
chapter = "chapter0_fundamentals"
section = "part0_prereqs"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part0_prereqs.tests as tests
from part0_prereqs.utils import display_array_as_img, display_soln_array_as_img

MAIN = __name__ == "__main__"

#%%
arr = np.load(section_dir / "numbers.npy")

display_array_as_img(arr[0, 0])


#%%
print(arr.shape)

#%%
arr1 = einops.rearrange(arr, "b c h w -> c (b h) w")
display_array_as_img(arr1)



#%%
arr2 = einops.repeat(arr[0], "c h w -> c (2 h) w")
display_array_as_img(arr2)

#%%
arr3 = einops.repeat(arr[0:2], "b c h w -> c (b h) (2 w)")
display_array_as_img(arr3)

#%%
arr4 = einops.repeat(arr[0], "c h w -> c (h 2) w")
display_array_as_img(arr4)

#%%
arr5 = einops.rearrange(arr[0], "c h w -> h (c w)")
display_array_as_img(arr5)

#%%
arr6 = einops.rearrange(arr, "(b1 b2) c h w -> c (b1 h) (b2 w)", b1=2)
display_array_as_img(arr6)

#%%
arr7 = einops.rearrange(arr[1], "c w h -> c h w")
display_array_as_img(arr7)

#%%
arr8 = einops.reduce(
    arr, 
    "(b1 b2) c (h h2) (w w2) -> c (b1 h) (b2 w)", 
    "max",
    h2=2, 
    w2=2, 
    b2=3,
)
display_array_as_img(arr8)

##%%
# Broadcasting

# %%
