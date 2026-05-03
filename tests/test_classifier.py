import pytest
from fleet.classifier import TaskClassifier


def test_keyword_classify_code():
    clf = TaskClassifier()
    tag, conf = clf.classify("write a python function to sort a list")
    assert tag == "code"
    assert conf >= 0.85


def test_keyword_classify_creative():
    clf = TaskClassifier()
    tag, conf = clf.classify("write a poem about the ocean")
    assert tag == "creative"
    assert conf >= 0.85


def test_keyword_classify_uncertain():
    clf = TaskClassifier()
    tag, conf = clf.classify("do something nice")
    assert conf < 0.8
