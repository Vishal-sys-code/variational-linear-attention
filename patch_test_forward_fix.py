with open("tests/test_forward.py", "r") as f:
    code = f.read()

search = """            # Update S
            # S_t = S_{t-1} + v_t alpha_t^T
            S = S + torch.outer(v_t, alpha_t)"""

replace = """            # Update S
            # S_t = S_{t-1} + v_t alpha_t^T
            v_t_f32 = v_t / (torch.norm(v_t) + 1e-6)
            S = S + torch.outer(v_t_f32, alpha_t)"""

code = code.replace(search, replace)
with open("tests/test_forward.py", "w") as f:
    f.write(code)

print("Patched test_forward.py")
