import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

source = Path(__file__).resolve().parents[1] / "scripts/publish_release.py"
spec = importlib.util.spec_from_file_location("qwen35_publish_release", source)
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


class ReleaseTest(unittest.TestCase):
    def setUp(self):
        names = ["adapter-config.json", "LICENSE-model.txt", "NOTICE-model.txt",
                 "student.safetensors", "model.tar.gz"]
        files = {name: self.record(name.encode()) for name in names}
        self.manifest = "".join(f"{record['sha256']}  {name}\n" for name, record in files.items()).encode()
        files[release.CHECKSUMS] = self.record(self.manifest)
        self.request = {"tag": "v0.2.0", "release_id": 35, "source_commit": "a" * 40, "files": files}
        self.draft = {"id": 35, "draft": True, "tag_name": "v0.2.0",
                      "target_commitish": "a" * 40, "prerelease": True,
                      "html_url": "https://github.com/example/releases/tag/v0.2.0"}
        self.assets = [
            {"id": index, "name": name, "state": "uploaded", "size": record["bytes"],
             "digest": f"sha256:{record['sha256']}"}
            for index, (name, record) in enumerate(files.items(), start=1)
        ]

    @staticmethod
    def record(data):
        return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}

    def test_request_and_manifest_match(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "release.json"
            path.write_text(json.dumps(self.request))
            self.assertEqual(release.read_request(path), self.request)
        release.verify_checksums(self.manifest, self.request["files"])

    def test_manifest_must_list_every_reviewed_attachment(self):
        manifest = self.manifest.rsplit(b"\n", 2)[0] + b"\n"
        self.request["files"][release.CHECKSUMS] = self.record(manifest)
        with self.assertRaisesRegex(ValueError, "exact reviewed attachments"):
            release.verify_checksums(manifest, self.request["files"])

    def test_manifest_digest_refuses_changed_bytes(self):
        with self.assertRaisesRegex(ValueError, "reviewed bytes"):
            release.verify_checksums(self.manifest + b"\n", self.request["files"])

    def manifest_api_responses(self):
        assets = b"\n".join(json.dumps(asset).encode() for asset in self.assets)
        return assets, self.manifest

    def test_checks_record_and_attachments_before_publish(self):
        with (
            patch.object(release, "gh_json", return_value=self.draft),
            patch.object(release, "output", side_effect=self.manifest_api_responses()),
        ):
            self.assertEqual(release.inspect_release(self.request), self.draft)

    def test_unverified_attachment_stops_before_mutation(self):
        self.assets[0]["digest"] = "sha256:" + "0" * 64
        with (
            patch.object(release, "gh_json", return_value=self.draft) as api,
            patch.object(release, "output", side_effect=self.manifest_api_responses()),
            self.assertRaisesRegex(ValueError, "differs from the reviewed bytes"),
        ):
            release.publish(self.request)
        api.assert_called_once_with(f"{release.RELEASES}/35")

    def test_conflicting_tag_stops_before_mutation(self):
        with (
            patch.object(release, "inspect_release", return_value=self.draft),
            patch.object(release, "tagged_commit", return_value="b" * 40),
            patch.object(release, "gh_json") as api,
            self.assertRaisesRegex(ValueError, "reviewed source commit"),
        ):
            release.publish(self.request)
        api.assert_not_called()

    def test_publishes_and_verifies_reviewed_draft(self):
        public = dict(self.draft, draft=False)
        version = self.request["source_commit"]
        with (
            patch.object(release, "inspect_release", side_effect=[self.draft, public]) as inspect,
            patch.object(release, "tagged_commit", side_effect=[None, version, version]),
            patch.object(release, "gh_json") as api,
        ):
            self.assertEqual(release.publish(self.request), public["html_url"])
        self.assertEqual(inspect.call_count, 2)
        create_tag, publish_draft = api.call_args_list
        self.assertEqual(create_tag.args[1], "POST")
        self.assertEqual(create_tag.kwargs["payload"]["sha"], version)
        self.assertEqual(publish_draft.args[1], "PATCH")
        self.assertEqual(publish_draft.kwargs["payload"],
                         {"draft": False, "prerelease": True, "make_latest": "false"})

    def test_exact_already_public_release_is_idempotent(self):
        public = dict(self.draft, draft=False)
        with (
            patch.object(release, "inspect_release", return_value=public),
            patch.object(release, "tagged_commit", return_value=self.request["source_commit"]),
            patch.object(release, "gh_json") as api,
        ):
            self.assertEqual(release.publish(self.request), public["html_url"])
        api.assert_not_called()
