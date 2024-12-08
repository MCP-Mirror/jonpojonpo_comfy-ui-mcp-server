import asyncio
import json
import logging
import os
import uuid
import base64
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import aiohttp
import websockets
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (CallToolResult, ImageContent, TextContent, Tool,
                      EmbeddedResource)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("comfy-mcp-server")

@dataclass
class ComfyConfig:
    server_address: str
    client_id: str

class ComfyUIServer:
    def __init__(self):
        self.config = ComfyConfig(
            server_address=os.getenv("COMFY_SERVER", "127.0.0.1:8188"),
            client_id=str(uuid.uuid4())
        )
        self.app = Server("comfy-mcp-server")
        self.setup_handlers()

    def setup_handlers(self):
        @self.app.list_tools()
        async def list_tools() -> List[Tool]:
            """List available video generation tools."""
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
                            },
                            "negative_prompt": {
                                "type": "string",
                                "description": "Negative prompt describing what you don't want",
                                "default": "low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly"
                            },
                            "seed": {
                                "type": "number",
                                "description": "Seed for reproducible generation",
                                "default": 497797676867141
                            },
                            "width": {
                                "type": "number",
                                "description": "Video width in pixels",
                                "default": 768
                            },
                            "height": {
                                "type": "number",
                                "description": "Video height in pixels",
                                "default": 512
                            },
                            "num_frames": {
                                "type": "number",
                                "description": "Number of frames in the video",
                                "default": 97
                            },
                            "steps": {
                                "type": "number",
                                "description": "Number of sampling steps",
                                "default": 30
                            },
                            "fps": {
                                "type": "number",
                                "description": "Frames per second for the output video",
                                "default": 24
                            }
                        },
                        "required": ["prompt"]
                    }
                )
            ]

        @self.app.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent | ImageContent | EmbeddedResource]:
            """Handle tool execution for video generation."""
            if name != "generate_video":
                raise ValueError(f"Unknown tool: {name}")

            if not isinstance(arguments, dict) or "prompt" not in arguments:
                raise ValueError("Invalid generation arguments")

            try:
                logger.info(f"Generating video with arguments: {arguments}")
                video_data = await self.generate_video(
                    prompt=arguments["prompt"],
                    negative_prompt=arguments.get("negative_prompt", "low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly"),
                    seed=int(arguments.get("seed", 497797676867141)),
                    width=int(arguments.get("width", 768)),
                    height=int(arguments.get("height", 512)),
                    num_frames=int(arguments.get("num_frames", 97)),
                    steps=int(arguments.get("steps", 30)),
                    fps=int(arguments.get("fps", 24))
                )

                if video_data:
                    return [
                        ImageContent(
                            type="image",
                            data=base64.b64encode(video_data).decode('utf-8'),
                            mimeType="image/webp"  # Using WebP for animated content
                        )
                    ]
                else:
                    raise RuntimeError("No video data received")

            except Exception as e:
                logger.error(f"Generation error: {str(e)}")
                return [
                    TextContent(
                        type="text",
                        text=f"Video generation failed: {str(e)}"
                    )
                ]

    async def generate_video(
        self,
        prompt: str,
        negative_prompt: str,
        seed: int,
        width: int,
        height: int,
        num_frames: int,
        steps: int,
        fps: int
    ) -> bytes:
        """Generate a video using ComfyUI with LTX-V model."""
        # Construct ComfyUI workflow based on the provided JSON
        workflow = {
            "38": {
                "class_type": "CLIPLoader",
                "inputs": {},
                "widgets_values": ["t5xxl_fp16.safetensors", "ltxv"]
            },
            "44": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {},
                "widgets_values": ["ltx-video-2b-v0.9.safetensors"]
            },
            "6": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "clip": ["38", 0]
                },
                "widgets_values": [prompt]
            },
            "7": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "clip": ["38", 0]
                },
                "widgets_values": [negative_prompt]
            },
            "69": {
                "class_type": "LTXVConditioning",
                "inputs": {
                    "positive": ["6", 0],
                    "negative": ["7", 0]
                },
                "widgets_values": [25]
            },
            "70": {
                "class_type": "EmptyLTXVLatentVideo",
                "inputs": {},
                "widgets_values": [width, height, num_frames, 1]
            },
            "71": {
                "class_type": "LTXVScheduler",
                "inputs": {
                    "latent": ["70", 0]
                },
                "widgets_values": [steps, 2.05, 0.95, True, 0.1]
            },
            "72": {
                "class_type": "SamplerCustom",
                "inputs": {
                    "model": ["44", 0],
                    "positive": ["69", 0],
                    "negative": ["69", 1],
                    "sampler": ["73", 0],
                    "sigmas": ["71", 0],
                    "latent_image": ["70", 0]
                },
                "widgets_values": [True, seed, "randomize", 3]
            },
            "73": {
                "class_type": "KSamplerSelect",
                "inputs": {},
                "widgets_values": ["euler"]
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
                    "images": ["8", 0]
                },
                "widgets_values": ["ComfyUI", fps, False, 90, "default"]
            }
        }

        try:
            prompt_response = await self.queue_prompt(workflow)
            logger.info(f"Queued prompt, got response: {prompt_response}")
            prompt_id = prompt_response["prompt_id"]
        except Exception as e:
            logger.error(f"Error queuing prompt: {e}")
            raise

        uri = f"ws://{self.config.server_address}/ws?clientId={self.config.client_id}"
        logger.info(f"Connecting to websocket at {uri}")
        
        async with websockets.connect(uri) as websocket:
            while True:
                try:
                    message = await websocket.recv()
                    
                    if isinstance(message, str):
                        try:
                            data = json.loads(message)
                            logger.info(f"Received text message: {data}")
                            
                            if data.get("type") == "executing":
                                exec_data = data.get("data", {})
                                if exec_data.get("prompt_id") == prompt_id:
                                    node = exec_data.get("node")
                                    logger.info(f"Processing node: {node}")
                                    if node is None:
                                        logger.info("Generation complete signal received")
                                        break
                        except:
                            pass
                    else:
                        logger.info(f"Received binary message of length: {len(message)}")
                        if len(message) > 8:  # Check if we have actual video data
                            return message[8:]  # Remove binary header
                        else:
                            logger.warning(f"Received short binary message: {message}")
                
                except websockets.exceptions.ConnectionClosed as e:
                    logger.error(f"WebSocket connection closed: {e}")
                    break
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    continue

        raise RuntimeError("No valid video data received")

    async def queue_prompt(self, prompt: Dict[str, Any]) -> Dict[str, Any]:
        """Queue a prompt with ComfyUI."""
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"http://{self.config.server_address}/prompt",
                    json={
                        "prompt": prompt,
                        "client_id": self.config.client_id
                    }
                ) as response:
                    if response.status != 200:
                        text = await response.text()
                        raise RuntimeError(f"Failed to queue prompt: {response.status} - {text}")
                    return await response.json()
            except aiohttp.ClientError as e:
                raise RuntimeError(f"HTTP request failed: {e}")

async def main():
    """Main entry point for the ComfyUI MCP server."""
    server = ComfyUIServer()
    
    async with stdio_server() as (read_stream, write_stream):
        await server.app.run(
            read_stream,
            write_stream,
            server.app.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())