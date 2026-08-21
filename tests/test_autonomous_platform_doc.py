"""Test autonomous platform documentation module."""
import sys
sys.path.insert(0, 'src')


def test_autonomous_platform_doc():
    from cadgensis.autonomous_platform.documentation import DocumentationGenerator
    gen = DocumentationGenerator()
    assert gen is not None