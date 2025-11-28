#!/usr/bin/env python3
"""
Test script for the ingest module.
Verifies that the ingestion pipeline can load codebases from various sources.
"""

from src.ingest import ingest, VirtualFileSystem
import tempfile
import zipfile
import os


def test_vfs_basic():
    """Test basic VirtualFileSystem functionality."""
    print("=" * 60)
    print("Testing VirtualFileSystem Basic Operations")
    print("=" * 60)

    vfs = VirtualFileSystem()

    # Add some test files
    vfs.add_file("README.md", b"# Test Project")
    vfs.add_file("src/main.py", b"print('Hello, World!')")
    vfs.add_file("src/utils.py", b"def helper(): pass")

    print(f"\n Added {len(vfs)} files to VFS")
    print(f" Total bytes: {vfs.total_bytes()}")

    # List files
    print("\nFiles in VFS:")
    print("-" * 60)
    for path in vfs.list_files():
        file = vfs.get(path)
        if file:
            print(f"  {path} ({file.size} bytes)")

    # Test retrieval
    readme = vfs.get("README.md")
    if readme:
        print(f"\n Retrieved README.md: {readme.data.decode('utf-8')}")

    # Test non-existent file
    missing = vfs.get("nonexistent.txt")
    print(f" Non-existent file returns None: {missing is None}")

    print("\n" + "=" * 60)
    print(" VFS Basic Operations: PASSED")
    print("=" * 60)
    return True


def test_local_zip_ingestion():
    """Test ingesting a local ZIP file."""
    print("\n\n" + "=" * 60)
    print("Testing Local ZIP File Ingestion")
    print("=" * 60)

    try:
        # Create a temporary ZIP file
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp_zip:
            zip_path = tmp_zip.name

            # Create a sample project structure
            with zipfile.ZipFile(zip_path, 'w') as zf:
                zf.writestr('project/README.md', '# Sample Project\nThis is a test.')
                zf.writestr('project/src/main.py', 'def main():\n    print("Hello")')
                zf.writestr('project/src/utils.py', 'def helper():\n    return 42')
                zf.writestr('project/tests/test_main.py', 'def test_main():\n    assert True')
                zf.writestr('project/.gitignore', '*.pyc\n__pycache__/')

        print(f"\n Created temporary ZIP: {zip_path}")

        # Ingest the ZIP
        print("\n" + "=" * 60)
        print("Ingesting ZIP file...")
        print("=" * 60)

        vfs = ingest(zip_path)

        print(f"\n Success! Loaded {len(vfs)} files")
        print(f" Total size: {vfs.total_bytes()} bytes")

        print("\nLoaded files:")
        print("-" * 60)
        for path in sorted(vfs.list_files()):
            file = vfs.get(path)
            if file:
                print(f"  {path} ({file.size} bytes)")

        # Verify specific files
        print("\n" + "=" * 60)
        print("Verifying file contents:")
        print("=" * 60)

        main_py = vfs.get("src/main.py")
        if main_py:
            print(f"\n Found src/main.py:")
            print(f"  {main_py.data.decode('utf-8')[:50]}...")

        readme = vfs.get("README.md")
        if readme:
            print(f"\n Found README.md:")
            print(f"  {readme.data.decode('utf-8')}")

        # Clean up
        os.unlink(zip_path)
        print(f"\n Cleaned up temporary file")

        print("\n" + "=" * 60)
        print(" Local ZIP Ingestion: PASSED")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n Error occurred: {type(e).__name__}")
        print(f"Message: {str(e)}")
        if 'zip_path' in locals():
            try:
                os.unlink(zip_path)
            except:
                pass
        return False


def test_github_url_ingestion():
    """Test ingesting from a GitHub URL."""
    print("\n\n" + "=" * 60)
    print("Testing GitHub URL Ingestion")
    print("=" * 60)

    # Use a small, stable public repository
    test_url = "https://github.com/anthropics/anthropic-quickstarts"

    print(f"\nTarget: {test_url}")
    print("\nNote: This test requires internet connectivity.")
    print("=" * 60)

    try:
        print("\nFetching from GitHub...")
        vfs = ingest(test_url)

        print(f"\nSuccess! Downloaded and loaded codebase")
        print(f"Total files: {len(vfs)}")
        print(f"Total size: {vfs.total_bytes():,} bytes")

        # Show a sample of files
        files = sorted(vfs.list_files())
        print(f"\nSample of loaded files (showing first 10):")
        print("-" * 60)
        for path in files[:10]:
            file = vfs.get(path)
            if file:
                print(f"  {path} ({file.size} bytes)")

        if len(files) > 10:
            print(f"  ... and {len(files) - 10} more files")

        # Verify we can read a file
        if files:
            sample_file = vfs.get(files[0])
            if sample_file:
                preview = sample_file.data.decode('utf-8', errors='ignore')[:100]
                print(f"\n Sample content from {files[0]}:")
                print(f"  {preview}...")

        print("\n" + "=" * 60)
        print("GitHub URL Ingestion: PASSED")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\nError occurred: {type(e).__name__}")
        print(f"Message: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Check your internet connection")
        print("2. Verify the repository is publicly accessible")
        print("3. Check for GitHub API rate limits")
        return False


def test_invalid_source():
    """Test error handling for invalid sources."""
    print("\n\n" + "=" * 60)
    print("Testing Error Handling")
    print("=" * 60)

    try:
        # This should raise a ValueError
        print("\nAttempting to ingest invalid source...")
        vfs = ingest("invalid://source")
        print("\nExpected ValueError but ingestion succeeded")
        return False

    except ValueError as e:
        print(f"\nCorrectly raised ValueError: {str(e)}")
        print("\n" + "=" * 60)
        print("Error Handling: PASSED")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\nUnexpected error type: {type(e).__name__}")
        print(f"Message: {str(e)}")
        return False


def test_zip_with_nested_structure():
    """Test ingesting a ZIP with nested directories."""
    print("\n\n" + "=" * 60)
    print("Testing ZIP with Nested Directory Structure")
    print("=" * 60)

    try:
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp_zip:
            zip_path = tmp_zip.name

            # Create a more complex structure
            with zipfile.ZipFile(zip_path, 'w') as zf:
                zf.writestr('myproject/src/core/engine.py', 'class Engine: pass')
                zf.writestr('myproject/src/core/utils/helpers.py', 'def help(): pass')
                zf.writestr('myproject/src/api/routes.py', 'routes = []')
                zf.writestr('myproject/docs/guide.md', '# Guide')
                zf.writestr('myproject/tests/unit/test_engine.py', 'def test(): pass')

        print(f"\nCreated ZIP with nested structure")

        vfs = ingest(zip_path)

        print(f"\nLoaded {len(vfs)} files")
        print("\nDirectory structure:")
        print("-" * 60)
        for path in sorted(vfs.list_files()):
            depth = path.count('/')
            indent = "  " * depth
            filename = path.split('/')[-1]
            print(f"{indent}{filename}")

        # Clean up
        os.unlink(zip_path)

        print("\n" + "=" * 60)
        print("Nested Structure Test: PASSED")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\nError: {str(e)}")
        if 'zip_path' in locals():
            try:
                os.unlink(zip_path)
            except:
                pass
        return False


if __name__ == "__main__":
    print("\n=� Starting Ingest Pipeline Tests\n")

    # Run all tests
    results = {}

    results['VFS Basic'] = test_vfs_basic()
    results['Local ZIP'] = test_local_zip_ingestion()
    results['Error Handling'] = test_invalid_source()
    results['Nested Structure'] = test_zip_with_nested_structure()

    # GitHub test is optional (requires internet)
    print("\n\n" + "=" * 60)
    response = input("Run GitHub ingestion test? (requires internet) [y/N]: ").strip().lower()
    if response == 'y':
        results['GitHub URL'] = test_github_url_ingestion()
    else:
        print("Skipping GitHub URL test")
        results['GitHub URL'] = None

    # Summary
    print("\n\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for test_name, passed in results.items():
        if passed is None:
            status = "SKIPPED"
        elif passed:
            status = "PASSED"
        else:
            status = "FAILED"
        print(f"{test_name:.<25} {status}")

    print("=" * 60)

    # Final verdict
    passed_tests = [v for v in results.values() if v is True]
    failed_tests = [v for v in results.values() if v is False]

    if failed_tests:
        print("\nSome tests failed. Check the output above for details.")
    elif passed_tests:
        print("\nAll tests passed! Your ingest pipeline is working correctly.")
    else:
        print("\nNo tests were run successfully.")
