with open("tests/test_memory.py", "r") as f:
    code = f.read()

search = """        # Manual update
        # v_t: (B, d) -> (B, d, 1)
        # alpha_t: (B, d) -> (B, 1, d)
        update = torch.matmul(v_t.unsqueeze(2), alpha_t.unsqueeze(1))
        S_expected = S_prev + update"""

replace = """        # Manual update
        # v_t: (B, d) -> (B, d, 1)
        # alpha_t: (B, d) -> (B, 1, d)
        v_t_f32 = v_t / (torch.norm(v_t, dim=-1, keepdim=True) + 1e-6)
        update = torch.matmul(v_t_f32.unsqueeze(2), alpha_t.unsqueeze(1))
        S_expected = S_prev + update"""

code = code.replace(search, replace)

search2 = """        # Manual sum
        S_manual = torch.zeros(B, d, d)
        for t in range(T):
            update = torch.matmul(v_seq[t].unsqueeze(2), alpha_seq[t].unsqueeze(1))
            S_manual += update"""

replace2 = """        # Manual sum
        S_manual = torch.zeros(B, d, d)
        for t in range(T):
            v_t_f32 = v_seq[t] / (torch.norm(v_seq[t], dim=-1, keepdim=True) + 1e-6)
            update = torch.matmul(v_t_f32.unsqueeze(2), alpha_seq[t].unsqueeze(1))
            S_manual += update"""

code = code.replace(search2, replace2)

search3 = """    def test_stability_renorm():
        \"\"\"
        Test 4 — Stability test with renormalization
        Construct a sequence where v_t and alpha_t have moderately large values.
        Verify renormalization triggers and S_t is scaled down.
        \"\"\"
        d = 4
        threshold = 10.0
        manager = MemoryMatrixManager(d_model=d, enable_renorm=True, renorm_threshold=threshold)
    
        B = 1
        manager.reset(batch_size=B)
    
        # Create inputs that will definitely exceed threshold
        # Threshold is 10.
        # If we add v @ alpha^T such that norm > 10.
        # Let v = [10, 0...], alpha = [1, 0...] -> update is matrix with 10 at (0,0).
        # Norm is 10. If existing S was small, new S has norm >= 10.
    
        v = torch.zeros(B, d)
        v[0, 0] = 20.0 # Large value
        alpha = torch.zeros(B, d)
        alpha[0, 0] = 1.0"""

replace3 = """    def test_stability_renorm():
        \"\"\"
        Test 4 — Stability test with renormalization
        Construct a sequence where v_t and alpha_t have moderately large values.
        Verify renormalization triggers and S_t is scaled down.
        \"\"\"
        d = 4
        threshold = 10.0
        manager = MemoryMatrixManager(d_model=d, enable_renorm=True, renorm_threshold=threshold)
    
        B = 1
        manager.reset(batch_size=B)
    
        # Create inputs that will definitely exceed threshold
        # Threshold is 10.
        # Since v is normalized, the norm of the update is ||v|| * ||alpha|| = 1 * ||alpha|| = ||alpha||.
        # So we just need to make alpha very large.
    
        v = torch.zeros(B, d)
        v[0, 0] = 20.0 # Will be normalized
        alpha = torch.zeros(B, d)
        alpha[0, 0] = 20.0"""

code = code.replace(search3, replace3)

search4 = """    def test_stability_no_renorm():
        \"\"\"
        Test stability with renorm disabled.
        Norm should grow.
        \"\"\"
        d = 4
        threshold = 10.0
        manager = MemoryMatrixManager(d_model=d, enable_renorm=False, renorm_threshold=threshold)
    
        B = 1
        manager.reset(batch_size=B)
    
        v = torch.zeros(B, d)
        v[0, 0] = 20.0
        alpha = torch.zeros(B, d)
        alpha[0, 0] = 1.0
    
        stats = manager.update(v, alpha)
    
        assert stats['renorm_triggered'] == 0.0, "Renormalization should NOT have triggered"
        assert stats['norm_max'] == 20.0"""

replace4 = """    def test_stability_no_renorm():
        \"\"\"
        Test stability with renorm disabled.
        Norm should grow.
        \"\"\"
        d = 4
        threshold = 10.0
        manager = MemoryMatrixManager(d_model=d, enable_renorm=False, renorm_threshold=threshold)
    
        B = 1
        manager.reset(batch_size=B)
    
        v = torch.zeros(B, d)
        v[0, 0] = 20.0
        alpha = torch.zeros(B, d)
        alpha[0, 0] = 20.0
    
        stats = manager.update(v, alpha)
    
        assert stats['renorm_triggered'] == 0.0, "Renormalization should NOT have triggered"
        assert abs(stats['norm_max'] - 20.0) < 1e-4"""

code = code.replace(search4, replace4)

search5 = """    def test_mixed_batch_renorm():
        \"\"\"
        Test batch where one element needs renorm and another doesn't.
        \"\"\"
        d = 4
        threshold = 10.0
        manager = MemoryMatrixManager(d_model=d, enable_renorm=True, renorm_threshold=threshold)
    
        B = 2
        manager.reset(batch_size=B)
    
        v = torch.zeros(B, d)
        alpha = torch.zeros(B, d)
    
        # Batch 0: Large update (20)
        v[0, 0] = 20.0
        alpha[0, 0] = 1.0
    
        # Batch 1: Small update (1)
        v[1, 0] = 1.0
        alpha[1, 0] = 1.0"""

replace5 = """    def test_mixed_batch_renorm():
        \"\"\"
        Test batch where one element needs renorm and another doesn't.
        \"\"\"
        d = 4
        threshold = 10.0
        manager = MemoryMatrixManager(d_model=d, enable_renorm=True, renorm_threshold=threshold)
    
        B = 2
        manager.reset(batch_size=B)
    
        v = torch.zeros(B, d)
        alpha = torch.zeros(B, d)
    
        # Batch 0: Large update (20)
        v[0, 0] = 20.0
        alpha[0, 0] = 20.0
    
        # Batch 1: Small update (1)
        v[1, 0] = 1.0
        alpha[1, 0] = 1.0"""

code = code.replace(search5, replace5)

with open("tests/test_memory.py", "w") as f:
    f.write(code)

print("Patched test_memory.py")