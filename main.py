import torch
import re
import os
from torch import autocast
from tqdm.auto import tqdm
from PIL import Image
from Stable_Diffusion import Diffusion_Generator
from ipywidgets import widgets
from IPython.display import display, clear_output
from base import *
import numpy as np
import matplotlib.pyplot as plt
class Imagination(Generator):
    def __init__(self, 
                 save_cache=None,
                 generator="Stable Diffusion v1.4",
                 mode="Image-to-image",
                 access_token=None):
        super().__init__()
        self.prompt = ["$$ in dark environment", 
                        "low-light $$"] 
        Stable_Diffusion_list = ["Stable Diffusion v1.4",
                                 "Small Stable Diffusion v0"]
        if generator in Stable_Diffusion_list:
            if access_token is None:
                raise Exception("Access token not found")
            self.model = Diffusion_Generator(version=generator,
                                             save_dir=save_cache,
                                             access_token=access_token) 
            
        self.tmp = 0

    def forward(self, 
                prompt,
                image, 
                nums=1, 
                check=False,
                trial_name=None,
                choice_prompt=None,
                parameters=dict()):
        
        def on_Next_button_clicked(b):
            nonlocal output_image
            output_image = self.model(prompt=prompt,
                                        image=init_image,
                                        **parameters)
            nonlocal ax
            ax.imshow(np.array(output_image))
            plt.draw()
        def on_True_button_clicked(b):
            if output_image != None:
                output_image.save('./' + trial_name + '/' + trial_name + "_" + str(self.tmp) + '.jpg')
                self.tmp += 1
        
        if choice_prompt != None:
            prompt = re.sub(r'\$\$', prompt, self.prompt[choice_prompt]) 
        if 'strength' not in parameters:
            parameters['strength'] = 0.5
        if 'guidance_scale' not in parameters:    
            parameters['guidance_scale'] = 7.5
        
        init_image = Image.open(image)
        height, width = init_image.shape[0], init_image.shape[1]
        trial_name = 'Trail' if trial_name == None else trial_name
        isExist = os.path.exists('./' + trial_name + '/')
        if not isExist:
            os.makedirs('./' + trial_name + '/')
        else:
            raise TrialAlreadyExistsError(trial_name)
        if check == True:
            output_image = None
            name = ['True', 'Next']
            buttons = [widgets.Button(description=description) for description in name]
            buttons[0].on_click(on_True_button_clicked)
            buttons[1].on_click(on_Next_button_clicked)
            combined = widgets.HBox([items for items in  buttons])
            out = widgets.Output()
            self.tmp = 0
            display(widgets.VBox([combined, out]))
            fig, ax = plt.subplots()
        else:
            for iter in tqdm(range(nums)):
                output_image = self.model(prompt=prompt,
                                        image=init_image,
                                        **parameters)
                output_image.save('./' + trial_name + '/' + trial_name + "_" + str(iter) + '.jpg')
    



        
            