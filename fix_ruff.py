import os, re

def fix_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # S110 fix for try/except/pass
    content = re.sub(
        r'except Exception:\s+pass',
        r'except (RuntimeError, ValueError, TypeError) as e:\n                logger.warning(f"Error ignored: {e}")',
        content
    )

    # BLE001 fix for except Exception as e:
    content = re.sub(
        r'except Exception as e:',
        r'except (RuntimeError, ValueError, TypeError, OSError) as e:',
        content
    )

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

for root, _, files in os.walk("src"):
    for file in files:
        if file.endswith(".py"):
            fix_file(os.path.join(root, file))
