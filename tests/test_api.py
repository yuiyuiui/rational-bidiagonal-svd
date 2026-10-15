import json
import subprocess
import sys

import numpy as np
import pytest

from rational_bidiagonal_svd import selected_svd, BackendUnavailable, ResourceLimitError
from rational_bidiagonal_svd._compiler import inspect_backend


@pytest.fixture(scope='module')
def default_result(tmp_path_factory):
    return selected_svd([2., 1.], [.25], [1, 0], verify=True,
                        work_dir=tmp_path_factory.mktemp('coupled'))


@pytest.mark.slow
def test_default_against_independent_svd(default_result):
    r = default_result
    b = np.array([[2., .25], [0., 1.]])
    expected = np.linalg.svd(b, compute_uv=False)[[1, 0]]
    assert np.max(abs(r.values-expected)) < 1e-13
    assert np.linalg.norm(b@r.right-r.left*r.values) < 1e-13
    assert np.linalg.norm(b.T@r.left-r.right*r.values) < 1e-13
    assert np.max(abs(r.left.T@r.left-np.eye(2))) < 1e-13
    assert np.max(abs(r.right.T@r.right-np.eye(2))) < 1e-13
    assert r.verification['passed']
    assert r.config['resolved_profile'] == 'proof'
    assert r.config['backend']['precision_bits'] == r.config['kernel_bits']
    assert r.indices == (1, 0)
    assert not r.values.flags.writeable
    for direction in r.left_directions+r.right_directions:
        assert abs(float(direction.exact_squared_norm())-1) < 1e-14
        assert abs(np.linalg.norm(direction.to_float())-1) < 1e-14


def test_zero_no_backend(monkeypatch):
    monkeypatch.setenv('CXX', '/no/compiler/here')
    r = selected_svd([0., 0.], [0.], [1, 0], verify=True)
    assert r.values.tolist() == [0., 0.]
    assert np.array_equal(r.left, np.array([[0., 1.], [1., 0.]]))
    assert r.verification['passed']
    assert r.config['resolved_profile'] == 'exact_zero'
    assert r.config['backend'] is None


def test_accuracy_dimension_boundary():
    r = selected_svd(np.zeros(1024), np.zeros(1023), [1023], target_bits=16)
    assert r.right[1023, 0] == 1.
    with pytest.raises(ValueError, match='1/64'):
        selected_svd(np.zeros(1025), np.zeros(1024), [0], target_bits=16)


@pytest.mark.parametrize('d,e,indices,options', [
    ([], [], [0], {}), ([1, 2], [], [0], {}),
    ([1, np.inf], [0], [0], {}), ([1+1j], [], [0], {}),
    ([2**53+1], [], [0], {}), ([1], [], [True], {}),
    ([1, 2], [0], [0, 0], {}), ([1], [], [-1], {}),
    ([1], [], [0.0], {}), ([1], [], [0], {'target_bits':15}),
    ([1], [], [0], {'target_bits':True}), ([1], [], [0], {'profile':'guess'}),
    ([1], [], [0], {'kernel_bits':4096}),
    ([1], [], [0], {'profile':'diagnostic', 'kernel_bits':220224}),
    ([1], [], [0], {'timeout':-1}), ([1], [], [0], {'verify':'yes'}),
])
def test_invalid_inputs(d,e,indices,options):
    with pytest.raises(ValueError):
        selected_svd(d,e,indices,**options)


def test_missing_compiler_never_falls_back(monkeypatch, tmp_path):
    monkeypatch.setenv('CXX', '/no/compiler/here')
    with pytest.raises(BackendUnavailable, match=r'C\+\+17 compiler'):
        selected_svd([1.], [], [0], cache_dir=tmp_path)


def test_resource_guard_precedes_build(monkeypatch):
    monkeypatch.setenv('CXX', '/no/compiler/here')
    with pytest.raises(ResourceLimitError):
        selected_svd([1.,2.], [.1], [0,1], max_workspace_bytes=1)


def test_signed_repeated_explicit_accuracy():
    r = selected_svd([-2.,2.],[0.],[1,0],target_bits=16,profile='proof',verify=True)
    assert np.array_equal(r.values,[2.,2.])
    b=np.diag([-2.,2.])
    assert np.linalg.norm(b@r.right-r.left*r.values) == 0
    assert r.verification['passed']
    assert r.config['target_bits'] == 16


def test_packet_resource_admission():
    with pytest.raises(ResourceLimitError, match='Packet rank'):
        selected_svd([2.,1.],[.25],[0],profile='diagnostic',max_workspace_bytes=1024)


def test_backend_identity_mismatch(tmp_path):
    fake = tmp_path/'fake'
    fake.write_text('#!/bin/sh\nprintf \'{"precision_bits":4096,"source_sha256":"wrong","index_bits":63,"boost_version":108900}\\n\'\n')
    fake.chmod(0o700)
    with pytest.raises(BackendUnavailable, match='mismatch'):
        inspect_backend(fake, 220224, 'expected')


def test_cli_and_api_defaults_same(tmp_path):
    inp = tmp_path/'input.json'
    inp.write_text(json.dumps({'d':[0.,0.], 'e':[0.], 'indices':[1]}))
    proc = subprocess.run([sys.executable,'-m','rational_bidiagonal_svd.cli',str(inp),str(tmp_path/'out'),'--verify'],
                          capture_output=True,text=True,check=True)
    actual = json.loads(proc.stdout)
    expected = selected_svd([0.,0.],[0.],[1],verify=True)
    assert actual['config'] == expected.config
    assert actual['values'] == expected.values.tolist()


@pytest.mark.slow
def test_diagnostic_is_explicit_and_instance_checked(tmp_path):
    r = selected_svd([2.,1.],[.25],[0,1],profile='diagnostic',verify=True,work_dir=tmp_path)
    assert r.config['accuracy_basis'] == 'diagnostic'
    assert r.config['backend']['precision_bits'] == 4096
    assert r.verification['passed']
    assert np.linalg.norm(r.left.T@r.left-np.eye(2)) < 1e-13
