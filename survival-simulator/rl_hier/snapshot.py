"""Compressed worker snapshots, including simulator-owned RNG and render caches.

Uses a private pickler, never global copyreg or simulator modifications. Loading
must be restricted to trusted local checkpoints, like torch.load itself.
"""
import copyreg
import io
import pickle
import random
import time
import zlib
import numpy as np


def restore_surface(size, pixels, alpha, colorkey):
    import pygame
    surface = pygame.image.frombytes(pixels, size, "RGBA")
    surface.set_alpha(alpha)
    surface.set_colorkey(colorkey)
    return surface


def reduce_surface(surface):
    import pygame
    return restore_surface, (surface.get_size(), pygame.image.tobytes(surface, "RGBA"),
                             surface.get_alpha(), surface.get_colorkey())


def dumps(env):
    import pygame
    if env.active:
        raise RuntimeError("Snapshots require a completed macrodecision")
    stream = io.BytesIO()
    pickler = pickle.Pickler(stream, protocol=pickle.HIGHEST_PROTOCOL)
    pickler.dispatch_table = copyreg.dispatch_table.copy()
    pickler.dispatch_table[pygame.Surface] = reduce_surface
    pickler.dump(dict(env=env, python=random.getstate(), numpy=np.random.get_state(),
                      episode_elapsed=time.monotonic() - env.metrics.started))
    return zlib.compress(stream.getvalue(), level=1)


def loads(blob):
    payload = pickle.loads(zlib.decompress(blob))
    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    env = payload["env"]
    env.metrics.started = time.monotonic() - payload["episode_elapsed"]
    return env
