"""CLI entrypoints for RunPod lifecycle and launching the browser Director Studio."""

from __future__ import annotations

import argparse
import asyncio
import json

import uvicorn

from comfy_factory_mcp.cloud_providers import load_active_factory_state
from comfy_factory_mcp.config import FactorySettings
from comfy_factory_mcp.service import FactoryService
from comfy_factory_mcp.web_studio import create_web_studio_app


async def _cmd_up(args: argparse.Namespace) -> None:
    svc = FactoryService(FactorySettings())
    result = await svc.factory_up(
        gpu_filter=args.gpu_filter,
        max_price_hr=args.max_price_hr,
        bootstrap_models=not args.no_bootstrap_models,
    )
    print(json.dumps(result, indent=2))


async def _cmd_down() -> None:
    svc = FactoryService(FactorySettings())
    result = await svc.factory_down()
    print(json.dumps(result, indent=2))


def _cmd_status() -> None:
    settings = FactorySettings()
    state = load_active_factory_state(settings)
    if state is None:
        print(json.dumps({"active": False}))
    else:
        print(state.model_dump_json(indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(prog="nairobiwood-cli")
    sub = parser.add_subparsers(dest="command", required=True)

    p_up = sub.add_parser("up", help="Provision RunPod GPU pod, bootstrap over SSH, and open tunnel")
    p_up.add_argument("--gpu-filter", default="L40S|RTX 6000 Ada|A100")
    p_up.add_argument("--max-price-hr", type=float, default=1.69)
    p_up.add_argument(
        "--no-bootstrap-models",
        action="store_true",
        help="Skip background model download during pod boot",
    )

    sub.add_parser("down", help="Terminate RunPod GPU pod and close SSH tunnel")
    sub.add_parser("status", help="Show active factory status")
    sub.add_parser("studio", help="Launch Director Studio web server on 127.0.0.1:8190")

    args = parser.parse_args()
    if args.command == "up":
        asyncio.run(_cmd_up(args))
    elif args.command == "down":
        asyncio.run(_cmd_down())
    elif args.command == "status":
        _cmd_status()
    elif args.command == "studio":
        settings = FactorySettings()
        uvicorn.run(
            create_web_studio_app(settings=settings),
            host="127.0.0.1",
            port=settings.studio_http_port,
        )


if __name__ == "__main__":
    main()
