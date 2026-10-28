"""Exact simulator-state fingerprints with portable model asset references."""

from copy import deepcopy
from functools import lru_cache
import hashlib
from html import unescape
from pathlib import Path
import re


HASH_VERSION = 'asset-content-v1'
_ASSET_FILE = re.compile(r'(?P<prefix>\bfile\s*=\s*)(?P<quote>[\"\x27])(?P<path>.*?)(?P=quote)')
_XML_TAG = re.compile(r'<!--.*?-->|<!\[CDATA\[.*?\]\]>|<[^>]*>', re.DOTALL)


@lru_cache(maxsize=512)
def _asset_sha256(path, size, mtime_ns, ctime_ns):
    # File metadata invalidates the cache if an installed asset changes.
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            result.update(block)
    return result.hexdigest()


def portable_digest(snapshot):
    """Hash the complete reset state, replacing XML asset paths by content hashes.

    The simulator receives the original XML and state. Only this detached copy
    is normalized, so installation directories cannot change fixture matching.
    XML attributes, simulator arrays, RNG state, and controller state otherwise
    remain byte-for-byte inputs to the original fingerprint algorithm.
    """
    if not isinstance(snapshot, dict) or not isinstance(snapshot.get('model'), str):
        raise ValueError('A reset snapshot must include its model XML')
    normalized = deepcopy(snapshot)

    def replace_asset(match):
        filename = unescape(match['path'])
        path = Path(filename)
        if not path.is_absolute():
            raise ValueError(f'Model asset path must be absolute: {filename}')
        path = path.resolve(strict=True)
        stat = path.stat()
        checksum = _asset_sha256(str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        return f"{match['prefix']}{match['quote']}sha256:{checksum}{match['quote']}"

    def replace_tag(match):
        tag = match.group()
        return tag if tag.startswith('<!') else _ASSET_FILE.sub(replace_asset, tag)

    normalized['model'] = _XML_TAG.sub(replace_tag, normalized['model'])
    return digest(normalized)


def digest(value):
    import numpy as np
    h=hashlib.sha256()
    def visit(x):
        h.update(type(x).__name__.encode()+b'\0')
        if isinstance(x,np.ndarray):
            assert not x.dtype.hasobject
            h.update(x.dtype.str.encode());h.update(repr(x.shape).encode());h.update(np.ascontiguousarray(x).tobytes())
        elif isinstance(x,dict):
            for k in sorted(x):visit(k);visit(x[k])
        elif isinstance(x,(tuple,list)):
            h.update(str(len(x)).encode())
            for v in x:visit(v)
        else:h.update(repr(x).encode())
        h.update(b'\xff')
    visit(value);return h.hexdigest()
