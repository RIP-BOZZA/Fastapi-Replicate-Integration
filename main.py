from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import replicate
import os
from dotenv import load_dotenv


load_dotenv() 
app = FastAPI()


class ImageGenerationInput(BaseModel):
    prompt: str
    width:float
    height:float


@app.post("/generate-image")
async def generate_image(input_data: ImageGenerationInput):
    output = replicate.run(
    "levelsio/90s-anime-aesthetics:a9c9af2d6fba4072c73064b213d6588f2193624728999cf8bf1cc0911b51c708",
    input={
        "model": "dev",
        "prompt": input_data.prompt,
        "lora_scale": 1.1,
        "num_outputs": 1,
        "aspect_ratio": "16:9",
        "output_format": "jpg",
        "guidance_scale": 3.5,
        "output_quality": 90,
        "prompt_strength": 0.8,
        "extra_lora_scale": 1,
        "num_inference_steps": 28,
        "width":input_data.width,
        "height":input_data.height
    }
    )
    return {"image_url ":output[0].url}