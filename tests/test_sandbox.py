import pytest

from dma.solvers.code.sandbox import Sandbox

BOOT = "import pandas as pd\nX0 = 41\n"


@pytest.fixture(scope="module")
def sb(tmp_path_factory):
    s = Sandbox(bootstrap=BOOT, workdir=str(tmp_path_factory.mktemp("sb")), timeout_s=5)
    s.start()
    yield s
    s.close()


def test_state_value_and_stdout(sb):
    r = sb.run("y = X0 + 1\nprint('hello')\ny * 2")
    assert r.ok and r.stdout.strip() == "hello" and r.value == "84"
    assert sb.run("y").value == "42"                     # persists across cells
    sb.reset()
    assert "NameError" in sb.run("y").error              # reset clears question state
    assert sb.run("X0").value == "41"                    # but keeps bootstrap


def test_error_and_final_answer(sb):
    r = sb.run("1/0")
    assert not r.ok and "ZeroDivisionError" in r.error
    r = sb.run("df = pd.DataFrame({'a':['x','y'],'b':[1,2]})\nfinal_answer('table', df)\nprint('never')")
    assert r.final["type"] == "table"
    assert r.final["value"] == {"columns": ["a", "b"], "rows": [["x", 1], ["y", 2]]}
    assert "never" not in r.stdout


def test_guards(sb):
    assert "not allowed" in (sb.run("import subprocess\nsubprocess.run(['ls'])").error or "")
    assert "blocked" in (sb.run("open('/etc/dma_test_file','w').write('x')").error or "")
    assert sb.run("open('ok.txt','w').write('x')").ok      # inside work dir is fine


def test_timeout_restarts_kernel(sb):
    r = sb.run("import time\ntime.sleep(30)", timeout_s=1)
    assert r.timed_out and sb.restarts >= 1
    assert sb.run("X0").value == "41"                      # fresh kernel re-bootstrapped


def test_timeout_keeps_rebased_business(sb):
    assert sb.rebase("BIZ = 'b1'") is None
    sb.run("import time\ntime.sleep(30)", timeout_s=1)
    assert sb.run("BIZ").value == "'b1'"                   # respawned kernel replays the rebase
    sb.reset()
    assert sb.run("BIZ").value == "'b1'"
