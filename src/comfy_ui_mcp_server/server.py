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
            """List available image generation tools."""
            return [
                Tool(
                    name="generate_image",
                    description="Generate an image using ComfyUI",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "Positive prompt describing what you want in the image"
                            },
                            "negative_prompt": {
                                "type": "string",
                                "description": "Negative prompt describing what you don't want",
                                "default": "bad hands, bad quality"
                            },
                            "seed": {
                                "type": "number",
                                "description": "Seed for reproducible generation",
                                "default": 8566257
                            },
                            "width": {
                                "type": "number",
                                "description": "Image width in pixels",
                                "default": 512
                            },
                            "height": {
                                "type": "number",
                                "description": "Image height in pixels",
                                "default": 512
                            }
                        },
                        "required": ["prompt"]
                    }
                )
            ]

        @self.app.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent | ImageContent | EmbeddedResource]:
            """Handle tool execution for image generation."""
            if name != "generate_image":
                raise ValueError(f"Unknown tool: {name}")

            if not isinstance(arguments, dict) or "prompt" not in arguments:
                raise ValueError("Invalid generation arguments")

            try:
                logger.info(f"Generating image with arguments: {arguments}")
                image_data = await self.generate_image(
                    prompt=arguments["prompt"],
                    negative_prompt=arguments.get("negative_prompt", "bad hands, bad quality"),
                    seed=int(arguments.get("seed", 8566257)),
                    width=int(arguments.get("width", 512)),
                    height=int(arguments.get("height", 512))
                )

                if image_data:
                    return [
                        ImageContent(
                            type="image",
                            data=base64.b64encode(image_data).decode('utf-8'),
                            mimeType="image/png"
                        )
                    ]
                else:
                    raise RuntimeError("No image data received")

            except Exception as e:
                logger.error(f"Generation error: {str(e)}")
                return [
                    TextContent(
                        type="text",
                        text=f"Image generation failed: {str(e)}"
                    )
                ]

    async def generate_image(
        self,
        prompt: str,
        negative_prompt: str,
        seed: int,
        width: int,
        height: int
    ) -> bytes:
        """Generate an image using ComfyUI."""
        # Construct ComfyUI workflow
        workflow = {
            "4": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {
                    "ckpt_name": "v1-5-pruned-emaonly.safetensors"
                }
            },
            "5": {
                "class_type": "EmptyLatentImage",
                "inputs": {
                    "batch_size": 1,
                    "height": height,
                    "width": width
                }
            },
            "6": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "clip": ["4", 1],
                    "text": prompt
                }
            },
            "7": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "clip": ["4", 1],
                    "text": negative_prompt
                }
            },
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "cfg": 8,
                    "denoise": 1,
                    "latent_image": ["5", 0],
                    "model": ["4", 0],
                    "negative": ["7", 0],
                    "positive": ["6", 0],
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "seed": seed,
                    "steps": 20
                }
            },
            "8": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["3", 0],
                    "vae": ["4", 2]
                }
            },
            "save_image_websocket": {
                "class_type": "SaveImageWebsocket",
                "inputs": {
                    "images": ["8", 0]
                }
            },
            "save_image": {
                "class_type": "SaveImage",
                "inputs": {
                    "images": ["8", 0],
                    "filename_prefix": "mcp"
                }
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
                        if len(message) > 8:  # Check if we have actual image data
                            return message[8:]  # Remove binary header
                        else:
                            logger.warning(f"Received short binary message: {message}")
                
                except websockets.exceptions.ConnectionClosed as e:
                    logger.error(f"WebSocket connection closed: {e}")
                    break
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    continue

        raise RuntimeError("No valid image data received")

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