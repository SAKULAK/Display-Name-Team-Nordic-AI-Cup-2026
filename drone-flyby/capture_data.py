"""Optional exact-byte validation capture; deliberately stdlib-only."""

import base64
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import tempfile

logger = logging.getLogger(__name__)


def enabled(name):
    return os.environ.get(name, 'false').strip().lower() in ('true', '1', 'yes')


def safe_component(value):
    raw = str(value)
    safe = re.sub(r'[^A-Za-z0-9_-]', '_', raw)[:80] or 'empty'
    # Prefix Windows device names too. Hash changed IDs to avoid sanitization collisions.
    if safe.upper() in {'CON', 'PRN', 'AUX', 'NUL', *(f'COM{i}' for i in range(10)),
                        *(f'LPT{i}' for i in range(10))}:
        safe = '_' + safe
    if safe != raw:
        safe += '_' + hashlib.sha256(raw.encode()).hexdigest()[:12]
    return safe


def _atomic_write(path, data):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def capture_request(request):
    if not enabled('CAPTURE_ENABLED'):
        return
    try:
        view = request.view
        image = base64.b64decode(view.image, validate=True)
        digest = hashlib.sha256(image).hexdigest()
        metadata = dict(sequence_id=request.sequence_id, frame=request.frame,
            frame_index=request.frame_index, request_id=request.request_id,
            resolution_level=view.resolution_level, center_x=view.center_x, center_y=view.center_y,
            source_region_xyxy=list(view.source_region_xyxy), transmitted_width=view.width,
            transmitted_height=view.height, original_width=request.original_width,
            original_height=request.original_height, image_media_type=view.image_media_type,
            image_sha256=digest)
        if request.camera_command_feedback is not None:
            feedback = request.camera_command_feedback
            metadata['camera_command_feedback'] = feedback.model_dump(mode='json')
        encoded = json.dumps(metadata, indent=2, ensure_ascii=False).encode('utf-8')
        directory = (Path(os.environ.get('CAPTURE_ROOT', 'captures')) /
                     safe_component(os.environ.get('CAPTURE_RUN_ID', 'run')) /
                     safe_component(request.sequence_id))
        directory.mkdir(parents=True, exist_ok=True)
        stem = f'frame_{request.frame:06d}_L{view.resolution_level}_x{view.center_x}_y{view.center_y}'
        suffix = hashlib.sha256(str(request.request_id).encode()).hexdigest()[:16]
        attempt = 0
        while True:
            name = stem if attempt == 0 else f'{stem}_{suffix}_{attempt}'
            png, sidecar = directory / (name+'.png'), directory / (name+'.json')
            reservation = directory / (name+'.reserve')
            attempt += 1
            try:
                with reservation.open('x'):
                    pass
            except FileExistsError:
                continue
            try:
                if png.exists() or sidecar.exists():
                    continue
                _atomic_write(png, image)
                _atomic_write(sidecar, encoded)
                break
            finally:
                reservation.unlink(missing_ok=True)
        logger.info('CAPTURE frame=%s L%s region=%s bytes=%s sha256=%s',
                    request.frame, view.resolution_level, view.source_region_xyxy, len(image), digest)
    except Exception:
        logger.exception('CAPTURE failed frame=%s', getattr(request, 'frame', 'unknown'))
