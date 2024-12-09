import json
import logging
import os
from urllib import request
from dataclasses import dataclass
from typing import Any, Dict, List

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (TextContent, Tool)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("comfy-mcp-server")

@dataclass
class ComfyConfig:
    server_address: str

class ComfyUIServer:
    def __init__(self):
        self.config = ComfyConfig(
            server_address=os.getenv("COMFY_SERVER", "127.0.0.1:8188")
        )
        self.app = Server("comfy-mcp-server")
        self.setup_handlers()

    def setup_handlers(self):
        @self.app.list_tools()
        async def list_tools() -> List[Tool]:
            return [
                Tool(
                    name="generate_video",
                    description="Generate a video using LTX-V model in ComfyUI",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "Detailed positive prompt describing what you want in the video"
                            }
                        },
                        "required": ["prompt"]
                    }
                )
            ]

        @self.app.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            if name != "generate_video":
                return [TextContent(type="text", text=f"Unknown tool: {name}")]

            try:
                prompt = arguments.get("prompt", "A cinematic scene")
                self.queue_prompt(prompt)
                return [TextContent(type="text", text="Video generation queued successfully!")]
            except Exception as e:
                logger.error(f"Error: {str(e)}")
                return [TextContent(type="text", text=f"Error queueing generation: {str(e)}")]

    def queue_prompt(self, prompt: str) -> None:
        workflow = {
            "38": {
                "class_type": "CLIPLoader",
                "inputs": {
                    "clip_name": "t5xxl_fp16.safetensors",
                    "type": "ltxv"
                }
            },
            "44": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {
                    "ckpt_name": "ltx-video-2b-v0.9.safetensors"
                }
            },
            "6": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "clip": ["38", 0],
                    "text": prompt
                }
            },
            "7": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "clip": ["38", 0],
                    "text": "low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly"
                }
            },
            "69": {
                "class_type": "LTXVConditioning",
                "inputs": {
                    "positive": ["6", 0],
                    "negative": ["7", 0],
                    "frame_rate": 25
                }
            },
            "70": {
                "class_type": "EmptyLTXVLatentVideo",
                "inputs": {
                    "width": 768,
                    "height": 512,
                    "length": 97,
                    "batch_size": 1
                }
            },
            "71": {
                "class_type": "LTXVScheduler",
                "inputs": {
                    "latent": ["70", 0],
                    "steps": 100,
                    "max_shift": 2.05,
                    "base_shift": 0.95,
                    "terminal": 0.1,
                    "stretch": 0.1
                }
            },
            "73": {
                "class_type": "KSamplerSelect",
                "inputs": {
                    "sampler_name": "euler"
                }
            },
            "72": {
                "class_type": "SamplerCustom",
                "inputs": {
                    "model": ["44", 0],
                    "positive": ["69", 0],
                    "negative": ["69", 1],
                    "sampler": ["73", 0],
                    "sigmas": ["71", 0],
                    "latent_image": ["70", 0],
                    "noise_seed": 497797676867141,
                    "add_noise": True,
                    "cfg": 3.0
                }
            },
            "8": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["72", 0],
                    "vae": ["44", 2]
                }
            },
            "41": {
                "class_type": "SaveAnimatedWEBP",
                "inputs": {
                    "images": ["8", 0],
                    "fps": 24,
                    "filename_prefix": "ComfyUI",
                    "method": "default",
                    "lossless": False,
                    "quality": 90
                }
            }
        }

        data = json.dumps({"prompt": workflow}).encode('utf-8')
        req = request.Request(f"http://{self.config.server_address}/prompt", data=data)
        request.urlopen(req)

async def main():
    server = ComfyUIServer()
    async with stdio_server() as (read_stream, write_stream):
        await server.app.run(
            read_stream,
            write_stream,
            server.app.create_initialization_options()
        )

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())