"""Moved: every MRI step now has its own known-answer test.

    python -m sins_steps.run_tests                      # all steps + integration
    python -m sins_steps.step4_collapse.test            # one step
"""
from sins_steps.run_tests import main

if __name__ == "__main__":
    main()
