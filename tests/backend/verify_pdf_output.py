"""
Verification script for PDF report rendering.
Checks both normal and cold profiles generated from live API.
"""
import sys
import re

def main():
    print("=== CHECKING verify_fresh_report.pdf ===")
    with open("verify_fresh_report.pdf", "rb") as f:
        data = f.read().decode("latin-1", errors="ignore")

    targets = [
        ("Subsurface Temperature Structure", True),
        ("15 Standard Depths Matrix", True),
        ("128-D Latent Representation", True),
        ("Contextual Reference Points", True),
        ("OceanIQ observation archive", True),
        ("Depth Matrix Analysis", False),
        ("OSTIA", False),
        ("SMAP", False),
        ("CMEMS", False),
        ("OSCAR", False),
        ("ERA5", False),
        ("ASCAT", False),
    ]

    all_passed = True
    for term, should_exist in targets:
        count = data.count(term)
        passed = (count > 0) if should_exist else (count == 0)
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] '{term}' - Expected: {should_exist}, Found: {count}")
        if not passed:
            all_passed = False

    print("\n=== CHECKING verify_cold_report.pdf ===")
    with open("verify_cold_report.pdf", "rb") as f:
        cold_data = f.read().decode("latin-1", errors="ignore")

    cold_targets = [
        ("Not reached", True),
        ("0.0 m", False),
        ("0.0m", False),
    ]

    for term, should_exist in cold_targets:
        count = cold_data.count(term)
        passed = (count > 0) if should_exist else (count == 0)
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] Cold D26: '{term}' - Expected: {should_exist}, Found: {count}")
        if not passed:
            all_passed = False

    # Check page count
    pages = len(re.findall(r"/Type\s*/Page\b", data))
    print(f"\nFresh Report Page Count: {pages}")
    if pages != 4:
        print(f"[FAIL] Expected 4 pages, got {pages}")
        all_passed = False
    else:
        print("[PASS] Page count is exactly 4")

    if all_passed:
        print("\n>>> ALL PDF CHECKS PASSED SUCCESSFULLY! <<<")
        sys.exit(0)
    else:
        print("\n>>> SOME PDF CHECKS FAILED! <<<")
        sys.exit(1)

if __name__ == "__main__":
    main()
