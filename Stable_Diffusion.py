from base import Generator
import os
import torch
class Diffusion_Generator(Generator):
    def __init__(self, 
                 version, 
                 save_dir,
                 access_token):
        from diffusers import StableDiffusionImg2ImgPipeline
        from huggingface_hub import login
        login(token=access_token)
        cache_dir = save_dir if save_dir != None else './Stable_Diffusion/cache'
        if version == 'Stable Diffusion v1.4':
            model_path = "CompVis/stable-diffusion-v1-4"  
        if version == "Small Stable Diffusion v0":
            model_path = "OFA-Sys/small-stable-diffusion-v0/"
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                                            model_path,
                                            revision="fp16", 
                                            torch_dtype=torch.float16,
                                            use_auth_token=True,
                                            cache_dir=cache_dir,
                                        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipe = self.pipe.to(device)

    def forward(self, 
                prompt, 
                image, 
                **parameters):
        output_image = self.pipe(prompt=prompt, 
                                 init_image=image, 
                                 **parameters).images[0]
        return output_image

    
    