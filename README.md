# LT3SD: Latent Trees for 3D Scene Diffusion
### [Project Page](https://quan-meng.github.io/projects/lt3sd/) | [Paper](http://arxiv.org/abs/2409.08215) | [Video](https://youtu.be/AJ5sG9VyjGA)

![pipeline](assets/teaser.png)

We present LT3SD, a novel latent diffusion model for large-scale 3D scene generation. 

Recent advances in diffusion models have shown impressive results in 3D object generation, but are limited in spatial extent and quality when extended to 3D scenes. To generate complex and diverse 3D scene structures, we introduce a latent tree representation to effectively encode both lower-frequency geometry and higher-frequency detail in a coarse-to-fine hierarchy. We can then learn a generative diffusion process in this latent 3D scene space, modeling the latent components of a scene at each resolution level. 

To synthesize large-scale scenes with varying sizes, we train our diffusion model on scene patches and synthesize arbitrary-sized output 3D scenes through shared diffusion generation across multiple scene patches. Through extensive experiments, we demonstrate the efficacy and benefits of TL3SD for large-scale, high-quality unconditional 3D scene generation and for probabilistic completion for partial scene observations.


## Citation
If you find our work useful in your research, please consider citing:

	@misc{meng2024lt3sdlatenttrees3d,
		title={LT3SD: Latent Trees for 3D Scene Diffusion}, 
		author={Quan Meng and Lei Li and Matthias Nießner and Angela Dai},
		journal={arXiv preprint arXiv:2409.08215},
		year={2024}
	}
