# Releasing DataStore

This guide explains how to release a new version of DataStore to PyPI.

## Automated Release (Recommended)

DataStore uses GitHub Actions for automated testing, building, and publishing to PyPI.

### Prerequisites

1. **Maintainer access** to the GitHub repository
2. **PyPI Trusted Publisher** configured (see Configuration section below)

### Automated Release Process

#### 1. Prepare the Release

Update version number in `__init__.py`:
```python
__version__ = "0.2.0"  # Update to new version
```

Update `CHANGELOG.md` with release notes:
```markdown
## [0.2.0] - 2025-01-15

### Added
- New feature X
- New feature Y

### Fixed
- Bug fix Z
```

Commit changes:
```bash
git add __init__.py CHANGELOG.md
git commit -m "chore: bump version to 0.2.0"
git push origin master
```

#### 2. Create and Push Tag

Creating a tag will automatically trigger the build and publish process:

```bash
git tag -a v0.2.0 -m "Release version 0.2.0"
git push origin v0.2.0
```

#### 3. Monitor the Release

1. Go to https://github.com/auxten/chdb-ds/actions
2. Watch the "DataStore CI/CD" workflow complete
3. The workflow will:
   - Run all tests on Python 3.7-3.12
   - Build wheel and source distributions
   - Upload to PyPI automatically

#### 4. Create GitHub Release (Optional)

Once the package is on PyPI, create a GitHub release:

1. Go to https://github.com/auxten/chdb-ds/releases
2. Click "Draft a new release"
3. Select the tag you just created (v0.2.0)
4. Title: "chdb-ds v0.2.0"
5. Copy the CHANGELOG entry into the description
6. Publish release

### Configuring PyPI Trusted Publisher

Instead of using API tokens, we use PyPI's Trusted Publisher feature for secure publishing:

1. Go to https://pypi.org/manage/project/chdb-ds/settings/publishing/
2. Add a new publisher with:
   - **PyPI Project Name**: `chdb-ds`
   - **Owner**: `auxten`
   - **Repository name**: `chdb-ds`
   - **Workflow name**: `datastore-ci.yml`
   - **Environment name**: `pypi`
3. Save the configuration

No API tokens needed! GitHub will authenticate using OIDC.

---

## Manual Release (Alternative)

If you need to release manually, follow these steps.

## Prerequisites

1. **Maintainer access** to the PyPI project
2. **PyPI API token** configured in `~/.pypirc`
3. **Development environment** set up with all dependencies

## Manual Release Process

### 1. Prepare the Release

#### Update version number

Edit `__init__.py`:
```python
__version__ = "0.2.0"  # Update to new version
```

#### Update CHANGELOG.md

Add release notes under a new version section:
```markdown
## [0.2.0] - 2025-01-15

### Added
- New feature X
- New feature Y

### Fixed
- Bug fix Z
```

#### Commit changes

```bash
git add __init__.py CHANGELOG.md
git commit -m "chore: bump version to 0.2.0"
```

### 2. Run Tests

Ensure all tests pass before releasing:

```bash
# Run full test suite
make test

# Run with coverage
make test-coverage
```

### 3. Build Distribution

Clean previous builds and create new distribution files:

```bash
# Clean old builds
make clean

# Build source and wheel distributions
make build
```

This creates:
- `dist/chdb-ds-0.2.0.tar.gz` (source distribution)
- `dist/chdb_ds-0.2.0-py3-none-any.whl` (wheel distribution)

### 4. Test on TestPyPI (Optional but Recommended)

Upload to TestPyPI first to verify everything works:

```bash
make upload-test
```

Test installation from TestPyPI:

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple chdb-ds
```

### 5. Upload to PyPI

If TestPyPI installation works correctly, upload to production PyPI:

```bash
make upload
```

You'll be prompted for your PyPI username and password (or API token).

### 6. Create Git Tag

Tag the release in git:

```bash
git tag -a v0.2.0 -m "Release version 0.2.0"
git push origin v0.2.0
```

### 7. Create GitHub Release

1. Go to https://github.com/auxten/chdb-ds/releases
2. Click "Draft a new release"
3. Select the tag you just created (v0.2.0)
4. Title: "chdb-ds v0.2.0"
5. Copy the CHANGELOG entry into the description
6. Attach the distribution files from `dist/`
7. Publish release

## PyPI Configuration

### Setting up API Token

1. Go to https://pypi.org/manage/account/token/
2. Create a new API token with "Upload packages" scope
3. Save it to `~/.pypirc`:

```ini
[pypi]
username = __token__
password = pypi-AgEIcHlwaS5vcmcC...your-token...

[testpypi]
username = __token__
password = pypi-AgEIcHlwaS5vcmcC...your-test-token...
```

### Manual Upload (alternative to make commands)

```bash
# Install twine if not already installed
pip install twine

# Upload to TestPyPI
python -m twine upload --repository testpypi dist/*

# Upload to PyPI
python -m twine upload dist/*
```

## Versioning

DataStore follows [Semantic Versioning](https://semver.org/):

- **MAJOR** version (X.0.0): Incompatible API changes
- **MINOR** version (0.X.0): New functionality, backwards compatible
- **PATCH** version (0.0.X): Bug fixes, backwards compatible

Examples:
- `0.1.0` → `0.2.0`: Added new features (INSERT/UPDATE/DELETE)
- `0.2.0` → `0.2.1`: Fixed a bug in condition handling
- `1.0.0` → `2.0.0`: Changed core API (breaking changes)

## Post-Release Checklist

- [ ] Version bumped in `__init__.py`
- [ ] CHANGELOG.md updated
- [ ] All tests passing
- [ ] Package built successfully
- [ ] Uploaded to PyPI
- [ ] Git tag created and pushed
- [ ] GitHub release created
- [ ] Documentation updated (if needed)
- [ ] Announcement made (if major release)

## Troubleshooting

### Build fails with "No module named 'build'"

```bash
pip install build
```

### Upload fails with authentication error

Check your `~/.pypirc` configuration and ensure the API token is correct.

### Version already exists on PyPI

You cannot overwrite a version on PyPI. Bump the version number and try again.

### Wheel not building correctly

Ensure `setup.cfg` has:
```ini
[bdist_wheel]
universal = 0
```

This ensures Python 3 only wheels are built.

## Resources

- [Python Packaging User Guide](https://packaging.python.org/)
- [Semantic Versioning](https://semver.org/)
- [Keep a Changelog](https://keepachangelog.com/)
- [PyPI Help](https://pypi.org/help/)

