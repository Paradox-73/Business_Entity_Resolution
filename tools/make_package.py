"""Build the Round 2 submission package <team>_submission.zip in dist/.

  python tools/make_package.py [--team NAME] [--dist DIR] [--doc FILE] [--allow-other-output] [--validate]

Layout of the zip (problem statement, section "Final Submission Package"):

  output/matching_results.tsv                   <- output/matching_results.tsv (the file uploaded to the leaderboard)
  output/candidate_pairs.tsv                    <- output/candidate_pairs.tsv
  code/business_entity_resolution/README.md     <- code/business_entity_resolution/ (README.md, requirements.txt,
  code/business_entity_resolution/...              src/, sets/), without __pycache__, *.pyc and virtual environments
  code/business_entity_resolution/docs/         <- copies of the files the READMEs and the methodology cite:
                                                   EXPERIMENTS.md, submissions/LOG.md (as LOG.md),
                                                   submissions/<v>/finalize.json (as finalize/<v>.json, v9z-v10d),
                                                   docs/runbooks/*.md (as runbooks/)
  Documentation_template.md                     <- docs/METHODOLOGY.md (the filled-in methodology template)

sets/ holds parquet and pkl files that git ignores, so a fresh clone does not have them; this script stops if any is
missing or differs from the file the final build used (md5 list below, also in sets/README.md).

Checks before the zip is written:
  - output/ holds the final files: md5 equal to the uploaded v10d files (public leaderboard 0.990565), unless
    --allow-other-output is given;
  - the 14 files of sets/ exist with the md5 listed in SETS;
  - README.md, requirements.txt and src/ exist in code/business_entity_resolution/, and the cited files copied into
    its docs/ exist;
  - with --validate: the official validator (student_resource/utils/validate_submission.py --check-ids) passes on
    output/ against $BER_DATA/test (default student_resource/dataset/test).
After writing, the zip is read back and the CRC of every entry is checked (zipfile.testzip). The zip is first written
as <name>.part and renamed at the end, so a failed run leaves no partial zip under the final name.

Standard library only; run from any folder with Python 3.8 or newer.
"""
import argparse
import hashlib
import os
import subprocess
import sys
import time
import zipfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
PKG = os.path.join(ROOT, "code", "business_entity_resolution")

# md5 of the files uploaded on 27 Sep 2026 (submissions/v10d/; also checked by src/build_final.py)
OUTPUT = {
    "matching_results.tsv": "18412329111b5d9c43df3a58df0574d1",
    "candidate_pairs.tsv": "a2ddcb5d26ede94f780fd2c4d87519a7",
}
# md5 of the pair sets and the stacker model file that src/build_final.py applies (sets/README.md)
SETS = {
    "desc_veto_set.parquet": "0e65b2e78ba95252695cd81f99aeb492",
    "noise_add_set.parquet": "e8aad4b0cd80d661bfcd8c2dd00e263b",
    "fp_veto_set.parquet": "b5d41271195aa926b164995ccbae0a28",
    "recall_add_set.parquet": "4290d065039dcded471e3202f4578913",
    "moreveto_stem2_set.parquet": "36892522bd956551c00896991eb861ae",
    "tier12_removed.parquet": "44e0c2ffd59c4e1d506c7e4e16cdcab8",
    "tier12_restored.parquet": "dd2f090642a60861adcfeaf6108a8edd",
    "fr_typo_safe.parquet": "1c82f7d76b9f3ff85ad217cced772a97",
    "fr_same_address_safe.parquet": "d4cdb4d5f9ff31de933c124dc4f5cddc",
    "fr_amp_safe.parquet": "e434878696947eaf8bc228eb01affec9",
    "fr_street_veto.parquet": "97ecb200d6a23f1c370fa7a68fa936cc",
    "usi_stack_add.parquet": "6267908133292fd838da9acc44c3a2e0",
    "usi_stack_remove.parquet": "3ae1106081c1701c394b5df12e3a5c2a",
    "lgb_models_avg.pkl": "b1ef5f249461286b6e9cc46102d6eaeb",
}
SKIP_DIRS = {"__pycache__", ".ipynb_checkpoints", ".pytest_cache", ".venv", "venv", ".git"}
# files the READMEs and the methodology cite, copied into code/business_entity_resolution/docs/ of the zip
CITED_VERSIONS = ("v9z", "v9zm", "v10a", "v10b", "v10c", "v10d")
SKIP_SUFFIXES = (".pyc", ".pyo")
SKIP_NAMES = {".DS_Store", "Thumbs.db"}


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def fail(msg):
    print(f"ERROR: {msg}")
    sys.exit(1)


def package_files():
    """(absolute path, path inside the zip) for every file of code/business_entity_resolution/, sorted."""
    out = []
    for d, dirs, files in os.walk(PKG):
        # skip caches and any virtual environment (a folder holding pyvenv.cfg), e.g. a .venv made inside this folder
        dirs[:] = sorted(x for x in dirs if x not in SKIP_DIRS and not os.path.isfile(os.path.join(d, x, "pyvenv.cfg")))
        for f in sorted(files):
            if f in SKIP_NAMES or f.endswith(SKIP_SUFFIXES):
                continue
            p = os.path.join(d, f)
            out.append((p, "code/business_entity_resolution/" + os.path.relpath(p, PKG).replace(os.sep, "/")))
    return out


def cited_docs():
    """(absolute path, path inside the zip) of the repository files the READMEs and the methodology cite, which live
    outside code/business_entity_resolution/: they go to its docs/ folder in the zip."""
    dst = "code/business_entity_resolution/docs/"
    out = [(os.path.join(ROOT, "EXPERIMENTS.md"), dst + "EXPERIMENTS.md"),
           (os.path.join(ROOT, "submissions", "LOG.md"), dst + "LOG.md")]
    out += [(os.path.join(ROOT, "submissions", v, "finalize.json"), f"{dst}finalize/{v}.json") for v in CITED_VERSIONS]
    rb = os.path.join(ROOT, "docs", "runbooks")
    out += [(os.path.join(rb, f), dst + "runbooks/" + f) for f in sorted(os.listdir(rb)) if f.endswith(".md")]
    return out


def untracked(files):
    """Files about to be zipped that git does not track (sets/ data files excluded); empty when git is unavailable."""
    try:
        r = subprocess.run(["git", "-C", ROOT, "ls-files", "-z", "code/business_entity_resolution"],
                           capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return []
    tracked = {t for t in r.stdout.decode("utf8").split("\0") if t}
    sets_data = {"code/business_entity_resolution/sets/" + n for n in SETS}
    return [a for _, a in files
            if a.startswith("code/business_entity_resolution/") and a not in tracked and a not in sets_data]


def main():
    ap = argparse.ArgumentParser(description="Build <team>_submission.zip for the Round 2 package.")
    ap.add_argument("--team", default="Mommys_Good_Boys",
                    help="team name used in the file name (default: Mommys_Good_Boys)")
    ap.add_argument("--dist", default=os.path.join(ROOT, "dist"), help="output folder (default: <repo>/dist)")
    ap.add_argument("--doc", default=os.path.join(ROOT, "docs", "METHODOLOGY.md"),
                    help="methodology file stored as Documentation_template.md (default: docs/METHODOLOGY.md)")
    ap.add_argument("--allow-other-output", action="store_true",
                    help="package output/ even when it differs from the uploaded v10d files")
    ap.add_argument("--validate", action="store_true", help="run the official validator on output/ first")
    a = ap.parse_args()
    t0 = time.time()

    # 1. output/
    outdir = os.path.join(ROOT, "output")
    for name, want in OUTPUT.items():
        p = os.path.join(outdir, name)
        if not os.path.isfile(p):
            fail(f"{p} is missing; rebuild it with code/business_entity_resolution/src/build_final.py")
        got = md5(p)
        same = got == want
        print(f"output/{name:22s} md5 {got} {'= uploaded v10d' if same else '!= uploaded v10d ' + want}")
        if not same and not a.allow_other_output:
            fail("output/ is not the uploaded final file (use --allow-other-output to package it anyway)")

    # 2. sets/
    for name, want in SETS.items():
        p = os.path.join(PKG, "sets", name)
        if not os.path.isfile(p):
            fail(f"{p} is missing (git ignores the sets/ data files; copy them in, see sets/README.md)")
        if md5(p) != want:
            fail(f"{p}: md5 differs from the file the final build used ({want})")
    print(f"sets/: {len(SETS)} files, md5 as listed in sets/README.md")

    # 3. code folder and methodology document
    for need in ("README.md", "requirements.txt", "src"):
        if not os.path.exists(os.path.join(PKG, need)):
            fail(f"code/business_entity_resolution/{need} is missing")
    if not os.path.isfile(a.doc):
        fail(f"methodology file {a.doc} is missing")
    if os.path.exists(os.path.join(PKG, "docs")):
        fail("code/business_entity_resolution/docs/ exists; the zip's docs/ folder is filled from the repository files")
    docs = cited_docs()
    for src, _ in docs:
        if not os.path.isfile(src):
            fail(f"cited file {src} is missing")

    # 4. optional official validator
    if a.validate:
        data = os.environ.get("BER_DATA", os.path.join(ROOT, "student_resource", "dataset"))
        val = os.path.join(ROOT, "student_resource", "utils", "validate_submission.py")
        if not os.path.isfile(val):
            fail(f"validator not found: {val}")
        cmd = [sys.executable, val, "-m", os.path.join(outdir, "matching_results.tsv"),
               "-c", os.path.join(outdir, "candidate_pairs.tsv"), "-t", os.path.join(data, "test"), "--check-ids"]
        print("running the official validator ...", flush=True)
        if subprocess.run(cmd, env={**os.environ, "PYTHONIOENCODING": "utf8"}).returncode != 0:
            fail("the official validator did not pass")

    files = [(os.path.join(outdir, n), "output/" + n) for n in ("matching_results.tsv", "candidate_pairs.tsv")]
    files += package_files()
    extra = untracked(files)
    files += docs
    files.append((a.doc, "Documentation_template.md"))
    if extra:
        print(f"note: {len(extra)} packaged files are not tracked by git (commit them so the repository matches the zip):")
        for x in extra[:20]:
            print(f"  {x}")
        if len(extra) > 20:
            print(f"  ... and {len(extra) - 20} more")

    # 5. write, then read back
    os.makedirs(a.dist, exist_ok=True)
    dst = os.path.join(a.dist, f"{a.team}_submission.zip")
    part = dst + ".part"
    raw = 0
    with zipfile.ZipFile(part, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for src, arc in files:
            z.write(src, arc)
            raw += os.path.getsize(src)
    with zipfile.ZipFile(part) as z:
        bad = z.testzip()
        if bad is not None:
            fail(f"CRC check failed for {bad} in {part}")
        names = z.namelist()
    os.replace(part, dst)

    tops = {}
    for n in names:
        k = n.split("/")[0] if n.startswith("output/") or "/" not in n else "/".join(n.split("/")[:3])
        tops[k] = tops.get(k, 0) + 1
    print(f"wrote {dst}")
    print(f"  {len(names)} files; {raw / 1e6:.1f} MB before compression, {os.path.getsize(dst) / 1e6:.1f} MB zipped; "
          f"md5 {md5(dst)}")
    for k in sorted(tops):
        print(f"  {k}: {tops[k]} files")
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
