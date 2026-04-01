with open("tests/test_transformer.py", "r") as f:
    code = f.read()

code = code.replace("if \"lambda_net\" in name:", "if \"lambda_net\" in name or \"cls_head\" in name:")

with open("tests/test_transformer.py", "w") as f:
    f.write(code)
print("Patched!")
