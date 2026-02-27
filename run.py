import os
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from einops import rearrange
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
import zipfile
import io
import rembg
from fastapi import FastAPI, File, HTTPException, Response, UploadFile
import trimesh
from pygltflib import GLTF2

from src.utils.infer_util import save_video
from src.utils.train_util import instantiate_from_config
from src.utils.camera_util import (
    FOV_to_intrinsics, 
    get_zero123plus_input_cameras,
    get_circular_camera_poses,
)
from src.utils.mesh_util import save_obj, save_obj_with_mtl, save_glb
from datetime import datetime
from src.utils.infer_util import remove_background, resize_foreground


def convert_obj_to_glb_with_texture(obj_path, glb_path):
    # Load mesh with texture support
    mesh = trimesh.load(obj_path, force='mesh', process=False)

    # Export to glTF (.glb)
    export_data = mesh.export(file_type='glb')

    # Write to file
    with open(glb_path, 'wb') as f:
        f.write(export_data)

    return os.path.exists(glb_path)


def get_render_cameras(batch_size=1, M=120, radius=4.0, elevation=20.0, is_flexicubes=False):
    """
    Get the rendering camera parameters.
    """
    c2ws = get_circular_camera_poses(M=M, radius=radius, elevation=elevation)
    if is_flexicubes:
        cameras = torch.linalg.inv(c2ws)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    else:
        extrinsics = c2ws.flatten(-2)
        intrinsics = FOV_to_intrinsics(30.0).unsqueeze(0).repeat(M, 1, 1).float().flatten(-2)
        cameras = torch.cat([extrinsics, intrinsics], dim=-1)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1)
    return cameras


def render_frames(model, planes, render_cameras, render_size=512, chunk_size=1, is_flexicubes=False):
    """
    Render frames from triplanes.
    """
    frames = []
    for i in tqdm(range(0, render_cameras.shape[1], chunk_size)):
        if is_flexicubes:
            frame = model.forward_geometry(
                planes,
                render_cameras[:, i:i+chunk_size],
                render_size=render_size,
            )['img']
        else:
            frame = model.forward_synthesizer(
                planes,
                render_cameras[:, i:i+chunk_size],
                render_size=render_size,
            )['images_rgb']
        frames.append(frame)
    
    frames = torch.cat(frames, dim=1)[0]
    return frames


class ObjectModelGenerator:
    def __del__(self):
            """Clean up resources when object is deleted."""
            self.cleanup()

    def __init__(self, config_path, output_base_path='outputs/', device=None):
        """Initialize the 3D model generator.
        
        Args:
            config_path: Path to the model config YAML
            output_base_path: Base directory for outputs
            device: torch device to use (default: cuda if available)
        """
        self.config_path = config_path
        self.output_base_path = output_base_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.config = OmegaConf.load(config_path)
        self.config_name = os.path.basename(config_path).replace('.yaml', '')
        self.model_config = self.config.model_config
        self.infer_config = self.config.infer_config
        
        self.IS_FLEXICUBES = True if self.config_name.startswith('instant-mesh') else False

        self.image_path = os.path.join(self.output_base_path, self.config_name, 'images')
        self.mesh_path = os.path.join(self.output_base_path, self.config_name, 'meshes')
        self.video_path = os.path.join(self.output_base_path, self.config_name, 'videos')
        os.makedirs(self.image_path, exist_ok=True)
        os.makedirs(self.mesh_path, exist_ok=True)
        os.makedirs(self.video_path, exist_ok=True)

        self._init_models()

        self.rembg_session = rembg.new_session()
        print("3D Generation Initialized.")

    
    def cleanup(self):
        if hasattr(self, 'pipeline'):
            del self.pipeline
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'rembg_session'):
            del self.rembg_session

        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


    def _init_models(self):
        """Initialize the diffusion and reconstruction models."""
        print('Loading diffusion model...')

        local_diffusion_path = os.path.join("models", "models--sudo-ai--zero123plus-v1.2", "snapshots", "2da07e89919e1a130c9b5add1584c70c7aa065fd")
        custom_dir = "zero123plus"
        '''
        self.pipeline = DiffusionPipeline.from_pretrained(
            "sudo-ai/zero123plus-v1.2", 
            custom_pipeline=custom_dir,
            torch_dtype=torch.float16,
        )'''
        self.pipeline = DiffusionPipeline.from_pretrained(
            local_diffusion_path,
            custom_pipeline=custom_dir,
            torch_dtype=torch.float16,
        )


        self.pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.pipeline.scheduler.config, timestep_spacing='trailing'
        )
        
        # Load custom white-background UNet
        print('Loading custom white-background unet...')
        '''if os.path.exists(self.infer_config.unet_path):
            unet_ckpt_path = self.infer_config.unet_path
        else:
            unet_ckpt_path = hf_hub_download(
                repo_id="TencentARC/InstantMesh", 
                filename="diffusion_pytorch_model.bin", 
                repo_type="model"
            )'''
        unet_ckpt_path = os.path.join("models", "models--TencentARC--InstantMesh", "snapshots", "b785b4ecfb6636ef34a08c748f96f6a5686244d0", "diffusion_pytorch_model.bin")
        state_dict = torch.load(unet_ckpt_path, map_location='cpu')
        self.pipeline.unet.load_state_dict(state_dict, strict=True)
        self.pipeline = self.pipeline.to(self.device)
        
        # Load reconstruction model
        print('Loading reconstruction model...')
        self.model = instantiate_from_config(self.model_config)
        # If changing the model (base, large) and it does not exist locally, download first from HF
        '''if os.path.exists(self.infer_config.model_path):
            model_ckpt_path = self.infer_config.model_path
        else:
            model_ckpt_path = hf_hub_download(
                repo_id="TencentARC/InstantMesh", 
                filename=f"{self.config_name.replace('-', '_')}.ckpt", 
                repo_type="model"
            )'''
         
        model_ckpt_path = os.path.join("models", "models--TencentARC--InstantMesh", "snapshots", "b785b4ecfb6636ef34a08c748f96f6a5686244d0", "instant_mesh_large.ckpt")
        state_dict = torch.load(model_ckpt_path, map_location='cpu')['state_dict']
        state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith('lrm_generator.')}
        self.model.load_state_dict(state_dict, strict=True)
        self.model = self.model.to(self.device)

        if self.IS_FLEXICUBES:
            self.model.init_flexicubes_geometry(self.device, fovy=30.0)
        self.model = self.model.eval()

        print("Models successfully loaded from disk.")


    def generate(self, 
                input_image, 
                name=None, 
                diffusion_steps=75, 
                scale=1.0, 
                distance=4.5,
                views=6,
                remove_bg=True,
                export_texmap=False, 
                save_video_file=False,
                return_file_content=True,
                seed=42,
                format="obj"):
        """Generate a 3D model from an input image.
        
        Args:
            input_image: PIL Image or path to image file
            name: Name for the output files (default: auto-generated)
            diffusion_steps: Number of diffusion steps
            scale: Scale of generated object
            distance: Render distance
            views: Number of input views (4 or 6)
            remove_bg: Whether to remove background
            export_texmap: Whether to export texture map
            save_video_file: Whether to save video
            return_file_content: Return file content in memory rather than just paths
            seed: Random seed
            
        Returns:
            dict: Paths to generated files and/or file content
        """
        seed_everything(seed)
        
        # Process input image
        if isinstance(input_image, str):
            # It's a path
            name = name or os.path.basename(input_image).split('.')[0]
            input_image = Image.open(input_image)
        else:
            # Assume it's already a PIL Image
            name = name or f"generated_{seed}"
            
        # Remove background if requested
        if remove_bg:
            input_image = remove_background(input_image, self.rembg_session)
            input_image = resize_foreground(input_image, 0.85)
        
        # Generate multiview images
        print(f'Generating multiview images for {name}...')
        output_image = self.pipeline(
            input_image, 
            num_inference_steps=diffusion_steps, 
        ).images[0]

        output_image_path = os.path.join(self.image_path, f'{name}.png')
        output_image.save(output_image_path)
        
        # Convert to tensor
        images = np.asarray(output_image, dtype=np.float32) / 255.0
        images = torch.from_numpy(images).permute(2, 0, 1).contiguous().float()
        images = rearrange(images, 'c (n h) (m w) -> (n m) c h w', n=3, m=2)
        
        # Reconstruction
        print(f'Reconstructing 3D model for {name}...')
        input_cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0*scale).to(self.device)
        chunk_size = 20 if self.IS_FLEXICUBES else 1
        
        images = images.unsqueeze(0).to(self.device)
        images = v2.functional.resize(images, 320, interpolation=3, antialias=True).clamp(0, 1)

        if views == 4:
            indices = torch.tensor([0, 2, 4, 5]).long().to(self.device)
            images = images[:, indices]
            input_cameras = input_cameras[:, indices]
            
        result_paths = {
            'multiview_image': output_image_path
        }

        with torch.no_grad():
            # Get triplane
            planes = self.model.forward_planes(images, input_cameras)
            
            # Generate mesh
            mesh_extension = "obj" if export_texmap else "glb"
            mesh_path_idx = os.path.join(self.mesh_path, f'{name}.{mesh_extension}')

            mesh_out = self.model.extract_mesh(
                planes,
                use_texture_map=export_texmap,
                **self.infer_config,
            )

            vertices, faces, uvs, mesh_tex_idx, tex_map = mesh_out
            save_obj_with_mtl(
                vertices.data.cpu().numpy(),
                uvs.data.cpu().numpy(),
                faces.data.cpu().numpy(),
                mesh_tex_idx.data.cpu().numpy(),
                tex_map.permute(1, 2, 0).data.cpu().numpy(),
                mesh_path_idx,
            )

            print("[INFO] Mesh saved to", mesh_path_idx)

            if format == "glb":
                '''vertices, faces, vertex_colors = mesh_out
                save_glb(vertices, faces, vertex_colors, mesh_path_idx)'''
                glb_path = mesh_path_idx.replace('.obj', '.glb')
                convert_obj_to_glb_with_texture(mesh_path_idx, glb_path)
                mesh_path_idx = glb_path

            result_paths['mesh'] = mesh_path_idx
            
            # Generate video if requested
            if save_video_file:
                video_path_idx = os.path.join(self.video_path, f'{name}.mp4')
                render_size = self.infer_config.render_resolution
                render_cameras = get_render_cameras(
                    batch_size=1, 
                    M=120, 
                    radius=distance, 
                    elevation=20.0,
                    is_flexicubes=self.IS_FLEXICUBES,
                ).to(self.device)
                
                frames = render_frames(
                    self.model, 
                    planes, 
                    render_cameras=render_cameras, 
                    render_size=render_size, 
                    chunk_size=chunk_size, 
                    is_flexicubes=self.IS_FLEXICUBES,
                )
                
                save_video(
                    frames,
                    video_path_idx,
                    fps=30,
                )
                
                result_paths['video'] = video_path_idx

            if format == "obj":
                # Create a zip file in memory with all the model files
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    # Add the OBJ file
                    obj_filename = os.path.basename(mesh_path_idx)
                    zipf.write(mesh_path_idx, obj_filename)
                    
                    # If texture map was exported, add MTL and texture files
                    if export_texmap:
                        mtl_path = mesh_path_idx.replace('.obj', '.mtl')
                        if os.path.exists(mtl_path):
                            mtl_filename = os.path.basename(mtl_path)
                            zipf.write(mtl_path, mtl_filename)
                        
                        texture_path = mesh_path_idx.replace('.obj', '.png')
                        if os.path.exists(texture_path):
                            texture_filename = os.path.basename(texture_path)
                            zipf.write(texture_path, texture_filename)
                    
                    # Add video if it was created
                    if save_video_file and 'video' in result_paths:
                        video_filename = os.path.basename(result_paths['video'])
                        zipf.write(result_paths['video'], video_filename)
                    
                    # Add multiview image
                    image_filename = "multiview_" + os.path.basename(result_paths['multiview_image'])
                    zipf.write(result_paths['multiview_image'], image_filename)
                
                # Add the zip file content to results
                zip_buffer.seek(0)
                result_paths['model_content'] = zip_buffer.getvalue()
                result_paths['content_type'] = 'application/zip'
                result_paths['filename'] = f"{name}_3d_model.zip"
            else: 
                with open(mesh_path_idx, 'rb') as f:
                    mesh_content = f.read()

                result_paths['model_content'] = mesh_content
                result_paths['content_type'] = 'model/gltf-binary'
                result_paths['filename'] = f"{name}.glb"
            
            '''if return_file_content:
                with open(mesh_path_idx, 'rb') as f:
                    mesh_content = f.read()
                
                result_paths['model_content'] = mesh_content
                if export_texmap:
                    result_paths['content_type'] = 'application/octet-stream'
                    result_paths['filename'] = f"{name}.obj"
                else:
                    result_paths['content_type'] = 'model/gltf-binary'
                    result_paths['filename'] = f"{name}.glb"'''
            
        return result_paths
    

    def __del__(self):
        """Clean up resources."""
        # Free GPU memory when the object is deleted
        if hasattr(self, 'pipeline'):
            del self.pipeline
        if hasattr(self, 'model'):
            del self.model
        torch.cuda.empty_cache()


app = FastAPI(title="3D Model Generation Service")
MODEL_GENERATOR_CONFIG_PATH = "configs/instant-mesh-large.yaml"
#MODEL_GENERATOR_CONFIG_PATH = "configs/instant-mesh-base.yaml"

@app.post("/generate_3d_model/")
async def generate_3d_model(file: UploadFile = File(..., description="Tensor image file of the CH object to generate the 3D Model."),
                            format: str = "obj" # "obj" or "glb"
                            ):
    contents = await file.read()

    try:
        ch_object_image = Image.open(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image format.")
    
    ch_object_image = ch_object_image.convert("RGB")

    model_generator = ObjectModelGenerator(
        config_path=MODEL_GENERATOR_CONFIG_PATH,
        output_base_path="../outputs/"
    )   

    try:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_paths = model_generator.generate(
            input_image=ch_object_image,
            name=f"3DModelGenerated_{current_time}",
            export_texmap=True, #(format == "obj"),
            save_video_file=True,
            return_file_content=True,
            seed=42,
            format=format
        )

        if 'model_content' in result_paths:
            return Response(
                content=result_paths['model_content'],
                media_type=result_paths['content_type'],
                headers={
                    "Content-Disposition": f"attachment; filename={result_paths['filename']}"
                }
            )
    finally:
        model_generator.cleanup()




