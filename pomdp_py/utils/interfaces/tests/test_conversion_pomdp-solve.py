# Tests the conversion to .pomdp file format code

import sys
import pomdp_py
import subprocess
from pomdp_py.utils.test_utils import *
from pomdp_py.utils.interfaces.conversion import to_pomdp_file
from pomdp_problems.tiger.tiger_problem import TigerProblem, State
import os
import io
import glob

def test_pomdp_file_conversion(pomdp_solve_path):
    print("[testing] test_pomdp_file_conversion")
    tiger = TigerProblem(0.15, State("tiger-left"),
                         pomdp_py.Histogram({State("tiger-left"): 0.5,
                                             State("tiger-right"): 0.5}))
    # Generate a .pomdp file
    filename = "./test_tiger.POMDP"
    to_pomdp_file(tiger.agent, filename, discount_factor=0.95)
    assert os.path.exists(filename)

    print("[testing] Running pomdp-solve on generated %s" % filename)
    proc = subprocess.Popen([pomdp_solve_path, "-pomdp", filename],
                            stdout=subprocess.PIPE)
    solution_found = False
    for line in io.TextIOWrapper(proc.stdout, encoding="utf-8"):
        if "Solution found" in line:
            solution_found = True
    assert solution_found, "Something wrong - generated tiger POMDP could not be solved"
    print("Pass.")

    # Delete the generated pomdp file
    os.remove(filename)

    # Delete pomdp-solve generated files
    remove_files("./*.pg")
    remove_files("./*.alpha")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("To run test, do: %s <pomdp-solver-path>" % sys.argv[0])
        exit(1)
    solver_path = sys.argv[1]
    test_pomdp_file_conversion(solver_path)