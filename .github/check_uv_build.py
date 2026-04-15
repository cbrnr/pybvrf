#!/usr/bin/env python

"""Check that uv_build in pyproject.toml is compatible with the latest uv version."""

import os
import re
import sys

from packaging.specifiers import SpecifierSet
from packaging.version import Version

with open("pyproject.toml") as f:
    content = f.read()

match = re.search(r'"(uv_build\s*[^"]+)"', content)
if not match:
    print("ERROR: Could not find uv_build requirement in pyproject.toml")
    sys.exit(1)

requirement = match.group(1)
print(f"Found requirement: {requirement}")

spec_str = requirement.replace("uv_build", "").strip()
specifiers = SpecifierSet(spec_str)

latest = Version(os.environ["UV_LATEST_VERSION"])
print(f"Latest uv version: {latest}")

if latest in specifiers:
    print(f"OK: '{requirement}' is compatible with uv {latest}.")
else:
    new_lower = str(latest)
    new_upper = f"{latest.major}.{latest.minor + 1}.0"
    new_requirement = f"uv_build >= {new_lower}, < {new_upper}"
    print(f"'{requirement}' is NOT compatible with uv {latest}. Updating to: {new_requirement}")
    new_content = content.replace(match.group(0), f'"{new_requirement}"')
    with open("pyproject.toml", "w") as f:
        f.write(new_content)
    sys.exit(2)
