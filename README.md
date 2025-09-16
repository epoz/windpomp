# Windpomp

Because you need something to fill that DAM system.

## Downloading images from sources

## CLIP processor

Listen to a predefined que on Redis for incoming image thumbnails in an endless blocking loop.
For each image received, calc the CLIP embedding, search for it in the database, and return the top n matches to another que.

## Bikidata worker

Also supported...
