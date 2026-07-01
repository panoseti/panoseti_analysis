import sys
from pathlib import Path


def custom_transformation(data: str) -> str:
    """
    STUDENTS: Add your custom function here!
    For example, you could apply a median subtraction or multiply by a factor.
    """
    # TODO: Modify this function
    return data + " | [Transformed]"


def main():
    if len(sys.argv) != 3:
        print("Usage: python demo_l0_to_l1.py <input_l0> <output_l1>")
        sys.exit(1)

    l0_path = Path(sys.argv[1])
    l1_path = Path(sys.argv[2])

    l1_path.mkdir(parents=True, exist_ok=True)

    # Read L0 data
    with open(l0_path / "data.txt") as f:
        raw_data = f.read().strip()

    # Apply transformation
    calibrated_data = custom_transformation(raw_data)

    # Write L1 data
    with open(l1_path / "data.txt", "w") as f:
        f.write(calibrated_data)

    print(f"Successfully calibrated {l0_path.name} -> {l1_path.name}")


if __name__ == "__main__":
    main()
