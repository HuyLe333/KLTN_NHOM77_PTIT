import os
import re
import base64
import zlib

base_dir = r"C:\Users\admin\AppData\Local\Programs\Python\Python312\Lib\site-packages\FiinQuantX"
files_to_decrypt = {
    r"helper\BasicInfor.py": "decrypted_BasicInfor.py"
}

for rel_path, out_name in files_to_decrypt.items():
    full_path = os.path.join(base_dir, rel_path)
    if not os.path.exists(full_path):
        print(f"File {full_path} does not exist!")
        continue
    
    with open(full_path, "rb") as f:
        content = f.read()

    # Find the start of the bytes literal payload
    start_idx = content.rfind(b"b'") + 2
    end_idx = content.rfind(b"'")

    payload = content[start_idx:end_idx]
    
    try:
        _ = lambda __ : zlib.decompress(base64.b64decode(__[::-1]))
        decompressed = _(payload)
        
        out_path = os.path.join(r"d:\Khóa luận tốt nghiệp\KLTN\scratch", out_name)
        with open(out_path, "wb") as out_f:
            out_f.write(decompressed)
        print("Success decrypted:", out_name)
    except Exception as e:
        print("Failed to decrypt:", out_name, e)
