from diffusers import StableDiffusionInpaintPipeline
import torch
from torch.cuda.amp import custom_bwd, custom_fwd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_key = "runwayml/stable-diffusion-inpainting"
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    model_key, 
    torch_dtype=torch.float16, 
    safety_checker=None
).to(device)

class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        gt_grad, = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None

def train_step(image, mask_image, prompt, negative_prompt, 
               num_inference_steps=5, strength=0.5, num_images_per_prompt=1,
               masked_image_latents=None, eta=0.0, guidance_scale = 7.5,):
    generator = torch.Generator(device)
    alphas = pipe.scheduler.alphas_cumprod.to(device)
    
    height = pipe.unet.config.sample_size * pipe.vae_scale_factor
    width = pipe.unet.config.sample_size * pipe.vae_scale_factor
    
    do_classifier_free_guidance = True
    
    # 2. Define call parameters
    batch_size = 1
    
    # 3. Encode input prompt
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
    prompt,
    device,
    1,
    do_classifier_free_guidance,
    negative_prompt,
    )
    
    if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            
    # 4. set timesteps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps, num_inference_steps = pipe.get_timesteps(
        num_inference_steps=num_inference_steps, strength=strength, device=device)

    # at which timestep to set the initial noise (n.b. 50% if strength is 0.5)
    latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
    # create a boolean to check if the strength is set to 1. if so then initialise the latents with pure noise
    is_strength_max = strength == 1.0

    # 5. Preprocess mask and image
    init_image = pipe.image_processor.preprocess(image, height=height, width=width)
    init_image = init_image.to(dtype=torch.float32)
    
    # 6. Prepare latent variables
    num_channels_latents = pipe.vae.config.latent_channels
    num_channels_unet = pipe.unet.config.in_channels
    return_image_latents = num_channels_unet == 4
    
    latents_outputs = pipe.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents=None,
        image=init_image,
        timestep=latent_timestep,
        is_strength_max=is_strength_max,
        return_noise=True,
        return_image_latents=return_image_latents,
    )
    
    if return_image_latents:
        latents, noise, image_latents = latents_outputs
    else:
        latents, noise = latents_outputs
        
    # 7. Prepare mask latent variables
    mask_condition = pipe.mask_processor.preprocess(mask_image, height=height, width=width)

    if masked_image_latents is None:
        masked_image = init_image * (mask_condition < 0.5)
    else:
        masked_image = masked_image_latents 
    
    mask, masked_image_latents = pipe.prepare_mask_latents(
        mask_condition,
        masked_image,
        batch_size * num_images_per_prompt,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        do_classifier_free_guidance,
    )

        
    # 8. Check that sizes of mask, masked image and latents match
    if num_channels_unet == 9:
        # default case for runwayml/stable-diffusion-inpainting
        num_channels_mask = mask.shape[1]
        num_channels_masked_image = masked_image_latents.shape[1]
        if num_channels_latents + num_channels_mask + num_channels_masked_image != pipe.unet.config.in_channels:
            raise ValueError(
                f"Incorrect configuration settings! The config of `pipeline.unet`: {pipe.unet.config} expects"
                f" {pipe.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
                f" `num_channels_mask`: {num_channels_mask} + `num_channels_masked_image`: {num_channels_masked_image}"
                f" = {num_channels_latents+num_channels_masked_image+num_channels_mask}. Please verify the config of"
                " `pipeline.unet` or your `mask_image` or `image` input."
            )
    elif num_channels_unet != 4:
        raise ValueError(
            f"The unet {pipe.unet.__class__} should have either 4 or 9 input channels, not {pipe.unet.config.in_channels}."
        )
        
    # 9. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, eta)
    num_warmup_steps = len(timesteps) - num_inference_steps * pipe.scheduler.order
    for i, t in enumerate(timesteps):
        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

        # concat latents, mask, masked_image_latents in the channel dimension
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        if num_channels_unet == 9:
            latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)

        # predict the noise residual
        noise_pred = pipe.unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=None,
            return_dict=False,
        )[0]
        
        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        w = lambda alphas: (((1 - alphas) / alphas) ** 0.5) 
        grad = w(alphas[t]) * (noise_pred - noise)
        
        loss = SpecifyGradient.apply(latents, grad)
        
        return loss