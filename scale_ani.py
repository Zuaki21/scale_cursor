import os
import io
import struct
from PIL import Image

# ---------------- RIFF/ANI utilities ----------------
def _pad2(n: int) -> int:
    return n + (n & 1)

def _read_chunks(data: bytes):
    pos = 12
    end = len(data)
    while pos + 8 <= end:
        fourcc = data[pos:pos+4]
        size = struct.unpack('<I', data[pos+4:pos+8])[0]
        payload_start = pos + 8
        payload_end = payload_start + size
        if payload_end > end:
            break
        yield fourcc, data[payload_start:payload_end], pos, size
        pos = _pad2(payload_end)

def _parse_ani(data: bytes):
    if len(data) < 12 or data[:4] != b'RIFF' or data[8:12] != b'ACON':
        raise ValueError('Not an ANI (RIFF ACON) file.')

    anih, rate, seq, frames, extras = None, None, None, [], []
    for fourcc, payload, _, _ in _read_chunks(data):
        if fourcc == b'anih':
            anih = struct.unpack('<9I', payload[:36])
        elif fourcc == b'rate':
            count = len(payload)//4
            rate = list(struct.unpack('<'+'I'*count, payload[:count*4]))
        elif fourcc == b'seq ':
            count = len(payload)//4
            seq = list(struct.unpack('<'+'I'*count, payload[:count*4]))
        elif fourcc == b'LIST' and payload[:4] == b'fram':
            p = 4
            while p + 8 <= len(payload):
                cid = payload[p:p+4]
                csz = struct.unpack('<I', payload[p+4:p+8])[0]
                cdata = payload[p+8:p+8+csz]
                if cid == b'icon':
                    frames.append(bytes(cdata))
                p = _pad2(p + 8 + csz)
        else:
            extras.append((fourcc, payload))

    if anih is None:
        raise ValueError('Missing anih chunk.')
    cbSize, nFrames, nSteps, iW, iH, iBitCount, nPlanes, iDispRate, bfAttr = anih
    if nFrames != len(frames):
        nFrames = len(frames)
    if seq is None:
        seq = list(range(nFrames))
    if rate is None:
        rate = [10] * len(seq)

    return {
        'header': dict(cbSize=cbSize, nFrames=nFrames, nSteps=len(seq),
                       iWidth=iW, iHeight=iH, iBitCount=iBitCount,
                       nPlanes=nPlanes, iDispRate=iDispRate, bfAttributes=bfAttr),
        'rate': rate, 'seq': seq, 'frames': frames, 'extras': extras
    }

def _write_chunk(fourcc: bytes, payload: bytes) -> bytes:
    return fourcc + struct.pack('<I', len(payload)) + payload + (b'\x00' if (len(payload) % 2) else b'')

def _build_ani(header, rate, seq, frames, extras):
    anih_payload = struct.pack(
        '<9I',
        36,
        header['nFrames'],
        header['nSteps'],
        header.get('iWidth', 0),
        header.get('iHeight', 0),
        header.get('iBitCount', 32),
        header.get('nPlanes', 1),
        header.get('iDispRate', 0),
        header.get('bfAttributes', 1),
    )
    chunks = []
    chunks.append(_write_chunk(b'anih', anih_payload))
    if rate: chunks.append(_write_chunk(b'rate', struct.pack('<'+'I'*len(rate), *rate)))
    if seq:  chunks.append(_write_chunk(b'seq ', struct.pack('<'+'I'*len(seq), *seq)))

    sub = b'fram'
    for fr in frames:
        sub += _write_chunk(b'icon', fr)
    chunks.append(_write_chunk(b'LIST', sub))
    for fourcc, payload in extras:
        chunks.append(_write_chunk(fourcc, payload))

    body = b''.join(chunks)
    riff_size = len(body) + 4
    return b'RIFF' + struct.pack('<I', riff_size) + b'ACON' + body

# ---------------- ICO/CUR helpers ----------------
def _parse_icodir_first_entry(b: bytes):
    if len(b) < 22:
        raise ValueError('Invalid ICO/CUR data.')
    reserved, typ, count = struct.unpack('<HHH', b[:6])
    if reserved != 0 or count < 1:
        raise ValueError('Invalid ICONDIR.')
    off = 6
    width = b[off] or 256
    height = b[off+1] or 256
    if typ == 2:
        hotspot_x, hotspot_y = struct.unpack('<HH', b[off+4:off+8])
    else:
        hotspot_x = hotspot_y = None
    return {'type': typ, 'width': width, 'height': height, 'hotspot': (hotspot_x, hotspot_y)}

def _ico_bytes_from_pillow(img, size):
    out = io.BytesIO()
    img.save(out, format='ICO', sizes=[size])
    return out.getvalue()

def _patch_ico_to_cur(ico_bytes: bytes, hotspot_xy):
    b = bytearray(ico_bytes)
    b[2:4] = struct.pack('<H', 2)
    off = 6
    x, y = hotspot_xy
    b[off+4:off+8] = struct.pack('<HH', x, y)
    return bytes(b)

def _scale_frame_preserve_hotspot(frame_bytes: bytes, target_size=(64,64)) -> bytes:
    info = _parse_icodir_first_entry(frame_bytes)
    img = Image.open(io.BytesIO(frame_bytes)).convert('RGBA')
    sw, sh = img.size
    dw, dh = target_size
    scaled = img.resize((dw, dh), Image.NEAREST)

    hotspot = (0, 0)
    out_is_cur = False
    if info['type'] == 2 and info['hotspot'] != (None, None):
        out_is_cur = True
        hx, hy = info['hotspot']
        hotspot = (round(hx * dw / sw), round(hy * dh / sh))

    ico = _ico_bytes_from_pillow(scaled, (dw, dh))
    return _patch_ico_to_cur(ico, hotspot) if out_is_cur else ico

# ---------------- Main ----------------
def scale_ani_folder(input_dir: str, size: int):
    target_size = (size, size)
    output_dir = input_dir.rstrip("\\/") + f"_scaled{size}"
    os.makedirs(output_dir, exist_ok=True)

    ani_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.ani')]
    if not ani_files:
        print('⚠️ .ani が見つかりませんでした。')
        return

    for name in ani_files:
        in_path = os.path.join(input_dir, name)
        out_path = os.path.join(output_dir, name)  # ファイル名はそのまま
        try:
            with open(in_path, 'rb') as f:
                data = f.read()
            ani = _parse_ani(data)

            ani['header']['iWidth'], ani['header']['iHeight'] = target_size
            ani['header']['nSteps']  = len(ani['seq'])
            ani['header']['nFrames'] = len(ani['frames'])

            new_frames = [_scale_frame_preserve_hotspot(fr, target_size) for fr in ani['frames']]
            out_data = _build_ani(ani['header'], ani['rate'], ani['seq'], new_frames, ani['extras'])

            with open(out_path, 'wb') as f:
                f.write(out_data)
            print(f'✅ {name} → {out_path}')

        except Exception as e:
            print(f'❌ {name} の処理でエラー: {e}')

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('使い方: python scale_ani_hotspot_final.py "<入力フォルダ>" [サイズ]')
        print('例: python scale_ani_hotspot_final.py "C:\\Cursors" 128')
    else:
        folder = sys.argv[1]
        size = int(sys.argv[2]) if len(sys.argv) >= 3 else 64
        scale_ani_folder(folder, size)
