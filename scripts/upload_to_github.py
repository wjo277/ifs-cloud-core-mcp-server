"""
GitHub Release Upload Script for Quantized Models

This script automates the process of uploading the quantized FastAI models
to GitHub releases using the GitHub CLI with device authentication.
"""

import subprocess
import sys
import json
import time
from pathlib import Path
from typing import Optional, List, Dict


class GitHubReleaseUploader:
    """Upload models to GitHub releases using GitHub CLI."""

    def __init__(self, repo: str = "graknol/ifs-cloud-core-mcp-server"):
        """Initialize the uploader."""
        self.repo = repo
        self.dist_dir = Path(__file__).parent.parent / "dist"

        # Model files to upload
        self.model_files = {
            "primary": {
                "file": "fastai_intent_classifier_quantized.pkl",
                "description": "Quantized FastAI Intent Classifier (Primary - 48% smaller, 22% faster)",
            },
            "fallback": {
                "file": "fastai_intent_classifier.pkl",
                "description": "Original FastAI Intent Classifier (Fallback - Full precision)",
            },
        }

    def check_github_cli(self) -> bool:
        """Check if GitHub CLI is installed and available."""
        try:
            result = subprocess.run(
                ["gh", "--version"], capture_output=True, text=True, check=True
            )
            print(f"✅ GitHub CLI found: {result.stdout.strip()}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ GitHub CLI not found!")
            print("\nPlease install GitHub CLI:")
            print("- Windows: winget install GitHub.cli")
            print("- macOS: brew install gh")
            print("- Linux: https://cli.github.com/manual/installation")
            return False

    def check_authentication(self) -> bool:
        """Check if user is authenticated with GitHub CLI."""
        try:
            result = subprocess.run(
                ["gh", "auth", "status"], capture_output=True, text=True, check=True
            )
            print("✅ Already authenticated with GitHub CLI")
            return True
        except subprocess.CalledProcessError:
            print("⚠️ Not authenticated with GitHub CLI")
            return False

    def authenticate(self) -> bool:
        """Authenticate with GitHub using device flow."""
        print("\n🔐 Authenticating with GitHub...")
        print("This will open a browser window for device authentication.")

        try:
            # Use device flow authentication
            subprocess.run(["gh", "auth", "login", "--web"], check=True)
            print("✅ Authentication successful!")
            return True

        except subprocess.CalledProcessError as e:
            print(f"❌ Authentication failed: {e}")
            return False

    def check_model_files(self) -> bool:
        """Check if model files exist and get their info."""
        print("\n📁 Checking model files...")

        all_exist = True
        total_size = 0

        for model_type, info in self.model_files.items():
            file_path = self.dist_dir / info["file"]

            if file_path.exists():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                total_size += size_mb
                print(f"   ✅ {info['file']} ({size_mb:.1f} MB)")
            else:
                print(f"   ❌ {info['file']} - Not found!")
                all_exist = False

        if all_exist:
            print(f"\n📊 Total upload size: {total_size:.1f} MB")

            if total_size > 100:
                print("⚠️ Large upload detected - this may take some time")
        else:
            print("\n❌ Some model files are missing!")
            print("Please run: uv run python scripts/prepare_model_release.py")

        return all_exist

    def get_existing_releases(self) -> List[Dict]:
        """Get list of existing releases."""
        try:
            result = subprocess.run(
                [
                    "gh",
                    "release",
                    "list",
                    "--repo",
                    self.repo,
                    "--json",
                    "name,tagName,isPrerelease",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            return json.loads(result.stdout)
        except Exception as e:
            print(f"⚠️ Could not fetch existing releases: {e}")
            return []

    def create_release(
        self, tag: str, title: str, notes: str, prerelease: bool = False
    ) -> bool:
        """Create a new GitHub release."""
        print(f"\n🚀 Creating release {tag}...")

        try:
            cmd = [
                "gh",
                "release",
                "create",
                tag,
                "--repo",
                self.repo,
                "--title",
                title,
                "--notes",
                notes,
            ]

            if prerelease:
                cmd.append("--prerelease")

            subprocess.run(cmd, check=True)
            print(f"✅ Release {tag} created successfully!")
            return True

        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create release: {e}")
            return False

    def upload_assets(self, tag: str) -> bool:
        """Upload model files to the release."""
        print(f"\n📤 Uploading assets to release {tag}...")

        success = True

        for model_type, info in self.model_files.items():
            file_path = self.dist_dir / info["file"]

            print(f"   Uploading {info['file']}...")

            try:
                subprocess.run(
                    [
                        "gh",
                        "release",
                        "upload",
                        tag,
                        str(file_path),
                        "--repo",
                        self.repo,
                        "--clobber",  # Overwrite if exists
                    ],
                    check=True,
                )

                print(f"   ✅ {info['file']} uploaded successfully")

            except subprocess.CalledProcessError as e:
                print(f"   ❌ Failed to upload {info['file']}: {e}")
                success = False

        return success

    def generate_release_notes(self, tag: str) -> str:
        """Generate release notes for the model release."""
        return f"""# IFS Cloud MCP Server {tag}

## 🚀 Quantized Model Release

This release features the new **quantized FastAI intent classifier** as the default model, delivering significant performance improvements:

### 📊 Performance Improvements
- **48% smaller file size** (121.1 MB → 63.0 MB)
- **22% faster inference** (22.4 ms → 17.4 ms)  
- **Same accuracy** as original model
- **Lower memory usage** (~62 MB reduction)

### 📦 Release Assets

**Primary Model (Recommended):**
- `fastai_intent_classifier_quantized.pkl` - Optimized quantized model for production use

**Fallback Model:**
- `fastai_intent_classifier.pkl` - Original full-precision model for compatibility

### 🔧 Usage

The quantized model is now the **default choice** for new installations. The system will automatically:
1. Download the quantized model first
2. Fall back to original model if needed
3. Provide seamless compatibility

### 💾 Manual Download

```bash
# Download quantized model (default)
python -m ifs_cloud_mcp_server.model_downloader

# Download original model  
python -m ifs_cloud_mcp_server.model_downloader --original
```

### 🎯 Benefits for Production

- **Faster startup** - Smaller model loads faster
- **Better response time** - 22% faster predictions
- **Lower resource usage** - Reduced memory footprint
- **Same search quality** - No accuracy degradation

## 📋 Migration Notes

Existing installations will automatically upgrade to the quantized model on next restart. No manual intervention required.

---

*Built with FastAI, PyTorch quantization, and comprehensive testing.*"""

    def interactive_release_flow(self) -> bool:
        """Interactive flow for creating and uploading release."""
        print("\n" + "=" * 60)
        print("🚀 GITHUB RELEASE UPLOAD WIZARD")
        print("=" * 60)

        # Get existing releases for reference
        existing_releases = self.get_existing_releases()
        if existing_releases:
            print("\n📋 Existing releases:")
            for release in existing_releases[:5]:  # Show last 5
                prerelease_mark = " (prerelease)" if release.get("isPrerelease") else ""
                print(f"   • {release['tagName']}{prerelease_mark}")

        # Get release details from user
        print("\n📝 Release Information:")

        # Suggest next version
        if existing_releases:
            last_tag = existing_releases[0]["tagName"]
            print(f"   Last release: {last_tag}")

        tag = input("   Enter release tag (e.g., v1.1.0): ").strip()
        if not tag:
            print("❌ Tag is required!")
            return False

        title = input(
            f"   Enter release title (default: 'IFS Cloud MCP Server {tag}'): "
        ).strip()
        if not title:
            title = f"IFS Cloud MCP Server {tag}"

        prerelease = input("   Is this a prerelease? (y/N): ").strip().lower() == "y"

        # Generate release notes
        notes = self.generate_release_notes(tag)
        print(f"\n📄 Generated release notes preview:")
        print("-" * 40)
        print(notes[:500] + "..." if len(notes) > 500 else notes)
        print("-" * 40)

        confirm = (
            input("\nProceed with release creation and upload? (Y/n): ").strip().lower()
        )
        if confirm == "n":
            print("❌ Release cancelled by user")
            return False

        # Create release
        if not self.create_release(tag, title, notes, prerelease):
            return False

        # Upload assets
        if not self.upload_assets(tag):
            print("⚠️ Release created but some assets failed to upload")
            return False

        print(f"\n🎉 SUCCESS! Release {tag} created and assets uploaded!")
        print(f"🔗 View release: https://github.com/{self.repo}/releases/tag/{tag}")

        return True

    def run(self) -> bool:
        """Run the complete upload process."""
        print("🚀 GitHub Release Upload for Quantized Models")
        print("=" * 50)

        # 1. Check GitHub CLI
        if not self.check_github_cli():
            return False

        # 2. Check authentication
        if not self.check_authentication():
            if not self.authenticate():
                return False

        # 3. Check model files
        if not self.check_model_files():
            return False

        # 4. Interactive release flow
        return self.interactive_release_flow()


def main():
    """Main entry point."""
    uploader = GitHubReleaseUploader()

    try:
        success = uploader.run()

        if success:
            print("\n✅ Upload completed successfully!")
            print("\n📋 Next steps:")
            print("1. Test download with new release")
            print("2. Update DEFAULT_TAG in model_downloader.py")
            print("3. Update documentation if needed")
        else:
            print("\n❌ Upload failed or was cancelled")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n❌ Upload cancelled by user (Ctrl+C)")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
