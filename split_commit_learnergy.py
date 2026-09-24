"""
Run from inside C:\\Users\\wafis\\Documents\\learnergy
    python split_commit_learnergy.py
Splits rt_variance_gaussian_rbm.py into a fix-only commit and a docs commit,
then commits the rest of the doc-only files. Does not push.
"""
import subprocess

def run(*args):
    r = subprocess.run(["git", *args], capture_output=True, text=True)
    if r.returncode != 0:
        print("FAILED:", args)
        print(r.stdout, r.stderr)
        raise SystemExit(1)
    return r.stdout

TARGET = "learnergy/models/temporal/rt_variance_gaussian_rbm.py"

# 1. Grab the current (fix+docs) working copy and the original HEAD copy.
final_bytes = open(TARGET, "rb").read()
head_bytes = subprocess.run(["git", "show", f"HEAD:{TARGET}"], capture_output=True).stdout

nl = b"\r\n" if b"\r\n" in head_bytes else b"\n"
old = b"        self.optimizer.step()" + nl + nl + b"        return total_mse" + nl
new = (b"        self.optimizer.step()" + nl + nl +
       b"        # Clamp sigma to prevent gradient-driven collapse toward 0." + nl +
       b"        with torch.no_grad():" + nl +
       b"            self.sigma.data.clamp_(min=0.1, max=10.0)" + nl + nl +
       b"        return total_mse" + nl)
assert head_bytes.count(old) == 1, "anchor text not found in HEAD version, aborting"
fix_only_bytes = head_bytes.replace(old, new)

# 2. Write fix-only version, commit it alone.
with open(TARGET, "wb") as f:
    f.write(fix_only_bytes)
run("add", TARGET)
run("commit", "-m", "fix: clamp sigma to stop collapse")

# 3. Write back the full final (fix+docs) version. Diff against new HEAD is now docs-only.
with open(TARGET, "wb") as f:
    f.write(final_bytes)

DOC_FILES = [
    TARGET,
    "learnergy/models/temporal/rt_gaussian_rbm.py",
    "learnergy/models/temporal/rtdbn.py",
    "learnergy/models/temporal/rtrbm.py",
    "learnergy/models/temporal/__init__.py",
    "tests/learnergy/models/temporal/test_rt_gaussian_rbm.py",
    "tests/learnergy/models/temporal/test_rt_variance_gaussian_rbm.py",
    "tests/learnergy/models/temporal/test_rtdbn.py",
    "tests/learnergy/models/temporal/test_rtrbm.py",
]
run("add", *DOC_FILES)
run("commit", "-m", "docs: trim comments and docstrings")

print("Done. Run 'git log --oneline -3' and 'git status' to check.")
