"""Publish a reviewed draft of the Qwen35 experimental model bundle."""

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

REPOSITORY = "HyperGAN/qwen35-particle-sliders"
RELEASES = f"repos/{REPOSITORY}/releases"
CHECKSUMS = "SHA256SUMS"
REQUIRED_FILES = {CHECKSUMS, "adapter-config.json", "LICENSE-model.txt", "NOTICE-model.txt"}


def read_request(path):
    request = json.loads(path.read_bytes())
    if not isinstance(request, dict) or set(request) != {"tag", "release_id", "source_commit", "files"}:
        raise ValueError("Expected a tag, release ID, source commit and reviewed files")
    if not isinstance(request["tag"], str) or not re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z]+(?:\.[0-9A-Za-z]+)*)?", request["tag"]):
        raise ValueError("Expected a version tag")
    if type(request["release_id"]) is not int or request["release_id"] <= 0:
        raise ValueError("Expected a positive GitHub release ID")
    if not isinstance(request["source_commit"], str) or not re.fullmatch(r"[0-9a-f]{40}", request["source_commit"]):
        raise ValueError("Expected a full commit SHA")
    files = request["files"]
    if not isinstance(files, dict) or not REQUIRED_FILES.issubset(files):
        raise ValueError("Expected checksums, adapter config and model notices")
    if not any(name.endswith(".safetensors") for name in files) or not any(name.endswith(".tar.gz") for name in files):
        raise ValueError("Expected reviewed model weights and the release archive")
    for name, spec in files.items():
        if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name)
                or not isinstance(spec, dict) or set(spec) != {"bytes", "sha256"}
                or type(spec["bytes"]) is not int or spec["bytes"] <= 0
                or not isinstance(spec["sha256"], str)
                or not re.fullmatch(r"[0-9a-f]{64}", spec["sha256"])):
            raise ValueError(f"Invalid attachment record for {name}")
    return request


def output(*args, input=None):
    return subprocess.check_output(args, input=input)


def gh_json(*args, payload=None):
    if payload is None:
        return json.loads(output("gh", "api", *args))
    return json.loads(output("gh", "api", *args, "--input", "-", input=json.dumps(payload).encode()))


def verify_source(source):
    kind = output("git", "cat-file", "-t", source).strip()
    if kind != b"commit":
        raise ValueError("Release source must be a commit")
    subprocess.run(["git", "merge-base", "--is-ancestor", source, "HEAD"], check=True)


def tagged_commit(tag):
    ref = f"refs/tags/{tag}"
    result = output("git", "ls-remote", f"https://github.com/{REPOSITORY}.git", ref, f"{ref}^{{}}")
    refs = {}
    for line in result.decode().splitlines():
        sha, name = line.split("\t")
        refs[name] = sha
    return refs.get(f"{ref}^{{}}", refs.get(ref))


def verify_checksums(contents, files):
    checksum_record = files[CHECKSUMS]
    if (len(contents) != checksum_record["bytes"]
            or hashlib.sha256(contents).hexdigest() != checksum_record["sha256"]):
        raise ValueError("SHA256SUMS differs from its reviewed bytes")
    found = {}
    for line in contents.decode("ascii").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64}) [ *]([A-Za-z0-9][A-Za-z0-9._-]*)", line)
        if match is None or match[2] in found:
            raise ValueError("Invalid or duplicate filename in SHA256SUMS")
        found[match[2]] = match[1]
    expected = {name: record["sha256"] for name, record in files.items() if name != CHECKSUMS}
    if found != expected:
        raise ValueError("SHA256SUMS does not cover the exact reviewed attachments")


def inspect_release(request):
    release_id = request["release_id"]
    record = gh_json(f"{RELEASES}/{release_id}")
    if (record["id"] != release_id or record["tag_name"] != request["tag"]
            or record["target_commitish"] != request["source_commit"] or not record["prerelease"]):
        raise ValueError("Release record differs from the reviewed tag, source or prerelease")
    pages = output(
        "gh", "api", "--paginate", f"{RELEASES}/{release_id}/assets?per_page=100",
        "--jq", ".[] | @json",
    )
    rows = [json.loads(line) for line in pages.splitlines()]
    assets = {row["name"]: row for row in rows}
    files = request["files"]
    if len(assets) != len(rows) or assets.keys() != files.keys():
        raise ValueError("Release attachments differ from the reviewed list")
    for name, asset in assets.items():
        spec = files[name]
        if (asset["state"] != "uploaded" or asset["size"] != spec["bytes"]
                or asset["digest"] != f"sha256:{spec['sha256']}"):
            raise ValueError(f"GitHub attachment differs from the reviewed bytes for {name}")
    manifest_id = assets[CHECKSUMS]["id"]
    contents = output(
        "gh", "api", "-H", "Accept: application/octet-stream", f"{RELEASES}/assets/{manifest_id}",
    )
    verify_checksums(contents, files)
    return record


def verify_tag(request, required=False):
    current = tagged_commit(request["tag"])
    if (current is not None or required) and current != request["source_commit"]:
        raise ValueError("Release tag does not resolve to the reviewed source commit")
    return current


def publish(request):
    record = inspect_release(request)
    current = verify_tag(request, required=not record["draft"])
    if not record["draft"]:
        return record["html_url"]
    if current is None:
        gh_json(
            "--method", "POST", f"repos/{REPOSITORY}/git/refs",
            payload={"ref": f"refs/tags/{request['tag']}", "sha": request["source_commit"]},
        )
    verify_tag(request, required=True)
    gh_json(
        "--method", "PATCH", f"{RELEASES}/{request['release_id']}",
        payload={"draft": False, "prerelease": True, "make_latest": "false"},
    )
    published = inspect_release(request)
    verify_tag(request, required=True)
    if published["draft"]:
        raise ValueError("Reviewed prerelease is still a draft")
    return published["html_url"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("validate", "audit", "publish"))
    parser.add_argument("--request", type=Path, default=Path("release.json"))
    args = parser.parse_args()
    request = read_request(args.request)
    verify_source(request["source_commit"])
    if args.operation == "audit":
        record = inspect_release(request)
        verify_tag(request, required=not record["draft"])
        print(f"Verified {len(request['files'])} attachments for {request['tag']}")
    if args.operation == "publish":
        if (os.environ.get("GITHUB_REPOSITORY") != REPOSITORY
                or os.environ.get("GITHUB_EVENT_NAME") != "push"
                or os.environ.get("GITHUB_REF") != "refs/heads/main"
                or output("git", "rev-parse", "HEAD").decode().strip() != os.environ.get("GITHUB_SHA")):
            raise ValueError("Publication requires a push to the release repository's main branch")
        print(publish(request))


if __name__ == "__main__":
    main()
