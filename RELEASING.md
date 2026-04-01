# Releasing Recode

## PyPI

1. Create the `recode` project on PyPI if it does not exist yet.
   Use the package name `recode-cli`.
2. Add a Trusted Publisher for:
   - owner/repo: `ever-oli/recode`
   - workflow: `.github/workflows/publish-pypi.yml`
   - environment: `pypi`
3. Bump the version in:
   - `pyproject.toml`
   - `recode/__init__.py`
4. Push a tag like `v0.1.0`.

The tag will trigger:

- `.github/workflows/release.yml` to attach `dist/*` to a GitHub release
- `.github/workflows/publish-pypi.yml` to publish to PyPI

## Homebrew

Update `ever-oli/homebrew-tap/Formula/recode.rb` after the tag is pushed so the formula points at the immutable GitHub source tarball for that release.

Formula template:

```ruby
class Recode < Formula
  include Language::Python::Virtualenv

  desc "Terminal spaced repetition for coding problems and reference solutions"
  homepage "https://github.com/ever-oli/recode"
  url "https://github.com/ever-oli/recode/archive/refs/tags/v0.1.0.tar.gz"
  sha256 "<fill-from-downloaded-tarball>"
  license "MIT"

  depends_on "python@3.12"

  def install
    virtualenv_install_with_resources
  end

  test do
    assert_match "problems_dir=", shell_output("#{bin}/recode --paths")
  end
end
```

To compute the SHA locally:

```bash
curl -L -o /tmp/recode-v0.1.0.tar.gz https://github.com/ever-oli/recode/archive/refs/tags/v0.1.0.tar.gz
shasum -a 256 /tmp/recode-v0.1.0.tar.gz
```
