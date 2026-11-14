#!/usr/bin/env python3
"""Compile and run deterministic C++ tests for DetectionLimiter/DetectionPermit.

Run from any directory:
    python3 /path/to/inference-cpp-split/scripts/test_detection_limiter.py

The C++ test executable is built in a temporary directory, not in the repo.
No HTTP server, ONNX model, or Dify installation is required.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path


TEST_SOURCE = r'''
#include "inspectflow/detection_store.hpp"

#include <condition_variable>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <thread>

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void test_capacity_and_release() {
    DetectionLimiter limiter(2);
    require(limiter.try_acquire(), "first permit should succeed");
    DetectionPermit first(limiter);

    {
        require(limiter.try_acquire(), "second permit should succeed");
        DetectionPermit second(limiter);
        require(!limiter.try_acquire(), "third permit should be refused");
    }  // second returns its permit

    require(limiter.try_acquire(), "permit should be reusable after release");
    DetectionPermit reused(limiter);
    require(!limiter.try_acquire(), "active count exceeded capacity");
}

void test_early_return_and_exception() {
    DetectionLimiter limiter(1);

    auto early_return = [&]() {
        require(limiter.try_acquire(), "early-return acquire failed");
        DetectionPermit permit(limiter);
        return;
    };
    early_return();
    require(limiter.try_acquire(), "early return did not release permit");
    limiter.release();

    try {
        require(limiter.try_acquire(), "exception-path acquire failed");
        DetectionPermit permit(limiter);
        throw std::runtime_error("intentional test exception");
    } catch (const std::runtime_error&) {
        // The guard must already have been destroyed during stack unwinding.
    }

    require(limiter.try_acquire(), "exception did not release permit");
    limiter.release();
}

void test_simultaneous_threads() {
    DetectionLimiter limiter(2);
    std::mutex state_mutex;
    std::condition_variable state_changed;
    int holding = 0;
    bool may_finish = false;
    bool worker_failed = false;

    auto worker = [&]() {
        if (!limiter.try_acquire()) {
            std::lock_guard<std::mutex> lock(state_mutex);
            worker_failed = true;
            state_changed.notify_all();
            return;
        }
        DetectionPermit permit(limiter);
        std::unique_lock<std::mutex> lock(state_mutex);
        ++holding;
        state_changed.notify_all();
        state_changed.wait(lock, [&] { return may_finish; });
    };

    std::thread first(worker);
    std::thread second(worker);
    {
        std::unique_lock<std::mutex> lock(state_mutex);
        state_changed.wait(lock, [&] { return holding == 2 || worker_failed; });
    }

    // Do not throw while threads are joinable: release and join them first.
    const bool refused_while_full = !limiter.try_acquire();
    if (!refused_while_full) limiter.release();
    {
        std::lock_guard<std::mutex> lock(state_mutex);
        may_finish = true;
    }
    state_changed.notify_all();
    first.join();
    second.join();

    require(!worker_failed, "worker could not acquire its permit");
    require(refused_while_full, "third concurrent request entered when full");
    require(limiter.try_acquire(), "first worker did not release permit");
    DetectionPermit after_first(limiter);
    require(limiter.try_acquire(), "second worker did not release permit");
    DetectionPermit after_second(limiter);
    require(!limiter.try_acquire(), "capacity changed after concurrent test");
}

int main() {
    try {
        test_capacity_and_release();
        std::cout << "PASS: capacity and release\n";
        test_early_return_and_exception();
        std::cout << "PASS: early return and exception cleanup\n";
        test_simultaneous_threads();
        std::cout << "PASS: simultaneous threads and recovery\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
'''


def main():
    project = Path(__file__).resolve().parent.parent
    source = project / "src" / "detection_store.cpp"
    include = project / "include"
    if not source.is_file():
        print(f"Missing C++ source: {source}", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory(prefix="inspectflow-limiter-test-") as temp:
        test_cpp = Path(temp) / "test_limiter.cpp"
        executable = Path(temp) / "test_limiter"
        test_cpp.write_text(TEST_SOURCE, encoding="utf-8")
        compiler = os.environ.get("CXX", "c++")
        command = [
            compiler, "-std=c++17", "-pthread", "-I", str(include),
            str(test_cpp), str(source), "-lsqlite3", "-o", str(executable),
        ]
        try:
            subprocess.run(command, check=True)
            return subprocess.run([str(executable)], check=False).returncode
        except FileNotFoundError as error:
            print(f"Missing compiler: {error}", file=sys.stderr)
            return 2
        except subprocess.CalledProcessError as error:
            print(f"C++ compilation failed (exit {error.returncode})", file=sys.stderr)
            return 2


if __name__ == "__main__":
    raise SystemExit(main())
