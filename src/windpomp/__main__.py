import sys, os, time, random, io, json, pickle
import clip, torch
from PIL import Image
import asyncio
import redis.asyncio as redis
import numpy as np
import ssl
import urllib.request
import duckdb
from bikidata.query import redis_worker as biki_worker

ssl._create_default_https_context = (
    ssl._create_unverified_context
)  # Allow unverified SSL for downloading models
import logging
import multiprocessing as mp

log = logging.getLogger("windpomp")
handler = logging.StreamHandler()
log.addHandler(handler)
DEBUG = os.environ.get("DEBUG", "1") == "1"
fmt = "%(levelname)-9s %(name)s %(asctime)s %(message)s"
datefmt = "%Y-%m-%d %H:%M:%S"
handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
if DEBUG:
    log.setLevel(logging.DEBUG)
    handler.setLevel(logging.DEBUG)
    log.debug("Debug logging enabled")
else:
    log.setLevel(logging.INFO)
    handler.setLevel(logging.INFO)

REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
redis_client = redis.Redis(host=REDIS_HOST)

model, preprocess = clip.load("ViT-B/32", device="cpu", jit=False)

CLIP_Q_NAME = "windpomp:clip"

CLIP_STORAGE = os.environ.get("CLIP_STORAGE", "clipdata")
if not os.path.isdir(CLIP_STORAGE):
    os.makedirs(CLIP_STORAGE, exist_ok=True)

CLIP_DB = os.environ.get("CLIP_DB", "clip_features.duckdb")


async def clip_worker():
    log.debug("Entering worker loop, using Redis")
    while True:
        _, serial_query = await redis_client.blpop(CLIP_Q_NAME)
        opts = json.loads(serial_query)
        query_ticket = opts.get("query_ticket")
        if not query_ticket:
            log.error("No query ticket found in query")
            continue

        _, contents = await redis_client.blpop(query_ticket)
        image_tensor = preprocess(Image.open(io.BytesIO(contents)))
        image_features = model.encode_image(
            torch.unsqueeze(image_tensor.to("cpu"), dim=0)
        )
        image_embeddings = image_features.cpu().detach().numpy().astype("float32")
        vecbuf = image_embeddings[0]

        db = duckdb.connect(CLIP_DB)
        matches = db.execute(
            "select path from clip_features order by array_distance(features, ?::FLOAT[512]) limit 5",
            (vecbuf,),
        ).fetchall()

        store_id = opts.get("store_id")
        if store_id:
            filepath = os.path.join(CLIP_STORAGE, f"{store_id}.npy")
            if os.path.isfile(filepath):
                log.debug("File %s already exists, skipping", filepath)
            else:
                with open(filepath, "wb") as f:
                    f.write(vecbuf)
                log.debug(f"Stored embedding to {filepath} from {query_ticket}")

        await redis_client.lpush(query_ticket, json.dumps(matches))
        log.debug("Processed query ticket %s", query_ticket)


class FileNotFoundError(Exception):
    pass


async def push_image_to_clip_queue(
    filepath: str, store_id: str = None, await_result: bool = False
):
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File {filepath} not found")
    img = Image.open(filepath)
    img.thumbnail((256, 256))
    byte_buffer = io.BytesIO()
    img.save(byte_buffer, format="PNG")
    a_bytes = byte_buffer.getvalue()
    query_ticket = f"{time.time()}-{random.randint(0,1000000)}"
    ticket = {"query_ticket": query_ticket, "timestamp": time.ctime()}
    if store_id:
        ticket["store_id"] = store_id

    await redis_client.lpush(query_ticket, a_bytes)
    await redis_client.lpush(CLIP_Q_NAME, json.dumps(ticket))
    if await_result:
        popresult = await redis_client.blpop(query_ticket, timeout=10)
        if popresult is None:
            raise TimeoutError("Query timed out")
        _, result = popresult
        return result

    return query_ticket


def usage():
    print("Usage: windpomp.py [clipworker|bikiworker]")
    sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        usage()

    if sys.argv[1] == "clipworker":
        asyncio.run(clip_worker())
        sys.exit(0)
    elif sys.argv[1] == "bikiworker":
        asyncio.run(biki_worker())
        sys.exit(0)
    usage()
