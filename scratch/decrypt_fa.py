with open(r"C:\Users\admin\AppData\Local\Programs\Python\Python312\Lib\site-packages\FiinQuantX\core\FundamentalAnalysis.py", "rb") as f:
    content = f.read()

# Find the start of the bytes literal payload: exec((_)(b'payload'))
# It starts after b' and ends before ')
# Let's find the last occurrence of b"b'"
start_idx = content.rfind(b"b'") + 2
end_idx = content.rfind(b"'")

payload = content[start_idx:end_idx]
print("Payload length:", len(payload))
print("First 20 bytes:", payload[:20])
print("Last 20 bytes:", payload[-20:])

_ = lambda __ : __import__('zlib').decompress(__import__('base64').b64decode(__[::-1]))
decompressed = _(payload)

out_path = r"d:\Khóa luận tốt nghiệp\KLTN\scratch\decrypted_FundamentalAnalysis.py"
with open(out_path, "wb") as out_f:
    out_f.write(decompressed)
print("Success!")
