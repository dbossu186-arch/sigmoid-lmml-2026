"""Rewrite a TorchScript zip without the optional *.debug_pkl records (keeps 64-byte data alignment via torch's own writer)."""
import sys, zipfile, torch
src, dst = sys.argv[1], sys.argv[2]
zin = zipfile.ZipFile(src)
names = [n for n in zin.namelist() if not n.endswith(".debug_pkl")]
root = names[0].split("/")[0]
w = torch._C.PyTorchFileWriter(dst)
for n in names:
    rec = n[len(root) + 1:]
    if rec in (".data/version", "version"): continue  # writer adds its own version record
    if rec == ".data/serialization_id": continue
    b = zin.read(n); w.write_record(rec, b, len(b))
w.write_end_of_file()
print("kept", len(names), "dropped", len(zin.namelist()) - len(names))
