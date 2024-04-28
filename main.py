import torch
import re
import os
from torch import autocast
from tqdm.auto import tqdm
from PIL import Image
from Stable_Diffusion import Diffusion_Generator
from base import *
class Imagination(Generator):
    def __init__(self, 
                 save_cache=None,
                 generator="Stable Diffusion v1.4",
                 mode="Image-to-image"):
        super.__init__()
        self.prompt = ["$$ in dark environment", 
                        "low-light $$"] 
        if generator == "Stable Diffusion v1.4":
            self.model = Diffusion_Generator(version=generator,
                                             save_dir=save_cache)
    
    def forward(self, 
                prompt,
                image, 
                nums=1, 
                trial_name=None,
                self_prompt=None,
                **parameters):
        
        if self_prompt != None:
            prompt = self_prompt
        else:
            prompt = re.sub('$$', prompt, self.prompt)
        init_image = Image.open(image)
        trial_name = 'Trail' if trial_name == None else trial_name
        isExist = os.path.exists('./' + trial_name + '/')
        if not isExist:
            os.makedirs('./' + trial_name + '/')
        else:
            raise TrialAlreadyExistsError(trial_name)

        for iter in range(nums):
            output_image = self.model(prompt=prompt,
                                      image=init_image,
                                      **parameters)
            output_image.save('./' + trial_name + '/' + trial_name + "_" + str(iter) + '.jpg')
        