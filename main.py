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
class Imagination(Generator):
    def __init__(self, 
                 save_cache=None,
                 generator="Stable Diffusion v1.4",
                 mode="Image-to-image",
                 access_token=None):
        super().__init__()
        self.prompt = ["$$ in dark environment", 
                        "low-light $$"] 
        Stable_Diffusion_list = ["Stable Diffusion v1.4"]
        if generator in Stable_Diffusion_list:
            self.model = Diffusion_Generator(version=generator,
                                             save_dir=save_cache,
                                             access_token=access_token)

    def forward(self, 
                prompt,
                image, 
                nums=1, 
                check=False,
                trial_name=None,
                choice_prompt=None,
                parameters=dict()):
        
        if choice_prompt != None:
            prompt = re.sub('$$', prompt, self.prompt[choice_prompt]) 
        if 'strength' not in parameters:
            parameters['strength'] = 0.5
        if 'guidance_scale' not in parameters:    
            parameters['guidance_scale'] = 7.5
        
        init_image = Image.open(image)
        trial_name = 'Trail' if trial_name == None else trial_name
        isExist = os.path.exists('./' + trial_name + '/')
        if not isExist:
            os.makedirs('./' + trial_name + '/')
        else:
            raise TrialAlreadyExistsError(trial_name)
        if check == True:
            name = ['True', 'False']
            buttons = [widgets.Button(description=description) for description in name]
            buttons[0].on_click(self.on_True_button_clicked)
            buttons[1].on_click(self.on_False_button_clicked)
            combined = widgets.HBox([items for items in  buttons])
            out = widgets.Output()
        for iter in tqdm(range(nums)):
            output_image = self.model(prompt=prompt,
                                      image=init_image,
                                      **parameters)
            if check == True:
                display(widgets.VBox[combined, out])
                display(output_image)
                self.con_run = 0
                while self.con_run == 0:
                    pass
                if self.con_run == 1:
                    output_image.save('./' + trial_name + '/' + trial_name + "_" + str(iter) + '.jpg')
                clear_output()
            else:
                output_image.save('./' + trial_name + '/' + trial_name + "_" + str(iter) + '.jpg')
    
    
    def on_True_button_clicked(self):
        self.con_run = 1
    def on_False_button_clicked(self):
        self.con_run = 2
            