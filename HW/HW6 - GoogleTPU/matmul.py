# =============================================================
# Množenje matrica na Google TPU — Google Colab
# =============================================================
# Pre pokretanja: Runtime > Change runtime type > TPU
# =============================================================

# ── 1. Instalacija JAX sa TPU podrškom ──────────────────────
import subprocess
result = subprocess.run(
    ["pip", "install", "jax[tpu]",
     "-f", "https://storage.googleapis.com/jax-releases/libtpu_releases.html",
     "-q"],
    capture_output=True, text=True
)
print(result.stdout[-500:] if result.stdout else "JAX instaliran.")
print(result.stderr[-300:] if result.returncode != 0 else "")

# ── 2. Provjera uređaja ──────────────────────────────────────
import jax
import jax.numpy as jnp
import numpy as np
import time

print("JAX verzija:", jax.__version__)
devices = jax.devices()
print(f"Broj dostupnih uređaja: {len(devices)}")
for i, d in enumerate(devices):
    print(f"  Uređaj {i}: {d}")

# Provjera da li je TPU dostupan
device_type = jax.devices()[0].device_kind
print(f"\nTip uređaja: {device_type}")
if "tpu" not in device_type.lower():
    print("UPOZORENJE: TPU nije detektovan! Provjeri Runtime > Change runtime type.")
else:
    print("✓ TPU uspješno detektovan.")

# ── 3. Jednostavno množenje matrica (JIT na jednom čipu) ─────
print("\n" + "="*60)
print("TEST 1: JIT množenje matrica (jedan TPU čip)")
print("="*60)

N = 1024  # Dimenzija matrice

# Kreiranje random matrica na CPU-u
key = jax.random.PRNGKey(42)
key1, key2 = jax.random.split(key)

# Explicit device placement — stavlja matricu na prvi TPU
A = jax.device_put(jax.random.normal(key1, (N, N), dtype=jnp.float32), devices[0])
B = jax.device_put(jax.random.normal(key2, (N, N), dtype=jnp.float32), devices[0])

print(f"Matrica A: {A.shape}, dtype={A.dtype}, device={A.devices()}")
print(f"Matrica B: {B.shape}, dtype={B.dtype}, device={B.devices()}")

# Definisanje JIT-kompajlirane funkcije za množenje
@jax.jit
def matmul_jit(a, b):
    return jnp.matmul(a, b)

# ── Warm-up (prvi poziv kompajlira, ostali izvršavaju) ───────
print("\nWarm-up (JIT kompajliranje)...")
t_warmup_start = time.time()
C_warmup = matmul_jit(A, B)
C_warmup.block_until_ready()  # VAŽNO: JAX je asinhroni, mora se čekati
t_warmup_end = time.time()
print(f"Warm-up vreme (kompajliranje + izvršavanje): {(t_warmup_end - t_warmup_start)*1000:.1f} ms")

# ── Benchmarking (10 iteracija, bez kompajliranja) ──────────
NUM_ITER = 10
times = []

for i in range(NUM_ITER):
    t_start = time.time()
    C = matmul_jit(A, B)
    C.block_until_ready()  # Čekamo da TPU završi
    t_end = time.time()
    times.append((t_end - t_start) * 1000)  # ms

avg_time = np.mean(times)
std_time = np.std(times)
print(f"\nRezultati ({NUM_ITER} iteracija, matrica {N}×{N}):")
print(f"  Prosečno vreme: {avg_time:.3f} ms ± {std_time:.3f} ms")
print(f"  Min: {min(times):.3f} ms, Max: {max(times):.3f} ms")
print(f"  GFLOP/s: {(2 * N**3) / (avg_time * 1e-3) / 1e9:.1f}")

# Provjera dimenzija rezultata
print(f"\nRezultat C = A × B: oblik={C.shape}, dtype={C.dtype}")

# ── 4. bfloat16 množenje (nativni TPU format) ────────────────
print("\n" + "="*60)
print("TEST 2: bfloat16 množenje (nativni TPU format)")
print("="*60)

A_bf16 = A.astype(jnp.bfloat16)
B_bf16 = B.astype(jnp.bfloat16)

@jax.jit
def matmul_bf16(a, b):
    return jnp.matmul(a, b)

# Warm-up
_ = matmul_bf16(A_bf16, B_bf16).block_until_ready()

times_bf16 = []
for i in range(NUM_ITER):
    t_start = time.time()
    C_bf16 = matmul_bf16(A_bf16, B_bf16)
    C_bf16.block_until_ready()
    t_end = time.time()
    times_bf16.append((t_end - t_start) * 1000)

avg_bf16 = np.mean(times_bf16)
print(f"bf16 prosečno vreme: {avg_bf16:.3f} ms ± {np.std(times_bf16):.3f} ms")
print(f"bf16 GFLOP/s: {(2 * N**3) / (avg_bf16 * 1e-3) / 1e9:.1f}")
print(f"Speedup bf16 vs float32: {avg_time / avg_bf16:.2f}×")

# ── 5. pmap: Paralelizacija na svim 8 TPU čipovima ───────────
print("\n" + "="*60)
print("TEST 3: pmap — paralelizacija po svim TPU čipovima")
print("="*60)

num_devices = len(jax.devices())
print(f"Broj uređaja: {num_devices}")

if num_devices > 1:
    # pmap zahteva da prva dimenzija bude broj uređaja
    # Svaki čip dobija jedan par matrica
    A_pmap = jnp.ones((num_devices, N, N), dtype=jnp.bfloat16)
    B_pmap = jnp.ones((num_devices, N, N), dtype=jnp.bfloat16)

    # pmap automatski distribuira na sve uređaje
    @jax.pmap
    def matmul_pmap(a, b):
        return jnp.matmul(a, b)

    # Warm-up
    _ = matmul_pmap(A_pmap, B_pmap)
    jax.effects_barrier()

    times_pmap = []
    for i in range(NUM_ITER):
        t_start = time.time()
        C_pmap = matmul_pmap(A_pmap, B_pmap)
        jax.effects_barrier()  # Čekamo sve čipove
        t_end = time.time()
        times_pmap.append((t_end - t_start) * 1000)

    avg_pmap = np.mean(times_pmap)
    print(f"pmap batch prosečno vreme ({num_devices} matrica): {avg_pmap:.3f} ms ± {np.std(times_pmap):.3f} ms")
    print(f"Vreme po matrici: {avg_pmap / num_devices:.3f} ms")
    print(f"Ukupni GFLOP/s ({num_devices} čipova): {(num_devices * 2 * N**3) / (avg_pmap * 1e-3) / 1e9:.1f}")

    print(f"\nOblik rezultata: {C_pmap.shape}  (svaki od {num_devices} čipova vratio {N}×{N})")
else:
    print("Samo jedan uređaj detektovan — pmap test preskočen.")

# ── 6. Skalabilnost: Benchmark po veličinama matrica ─────────
print("\n" + "="*60)
print("TEST 4: Skalabilnost po veličini matrice (bfloat16)")
print("="*60)

sizes = [128, 256, 512, 1024, 2048, 4096]
print(f"{'N':>6} | {'Vreme (ms)':>12} | {'GFLOP/s':>10}")
print("-" * 35)

for n in sizes:
    key_a, key_b = jax.random.split(jax.random.PRNGKey(n))
    a = jax.random.normal(key_a, (n, n), dtype=jnp.bfloat16)
    b = jax.random.normal(key_b, (n, n), dtype=jnp.bfloat16)

    # JIT kompajliranje za ovu veličinu
    @jax.jit
    def _matmul(x, y):
        return jnp.matmul(x, y)

    # Warm-up
    _ = _matmul(a, b).block_until_ready()

    # Merenje
    iters = 5
    t_start = time.time()
    for _ in range(iters):
        result = _matmul(a, b).block_until_ready()
    t_total = (time.time() - t_start) * 1000 / iters

    gflops = (2 * n**3) / (t_total * 1e-3) / 1e9
    print(f"{n:>6} | {t_total:>12.3f} | {gflops:>10.1f}")

# ── 7. Provjera tačnosti ─────────────────────────────────────
print("\n" + "="*60)
print("TEST 5: Provjera tačnosti (TPU vs NumPy CPU referenca)")
print("="*60)

n_check = 256  # Mala matrica za provjeru
a_np = np.random.randn(n_check, n_check).astype(np.float32)
b_np = np.random.randn(n_check, n_check).astype(np.float32)

# NumPy referenca na CPU
C_ref = np.matmul(a_np, b_np)

# JAX na TPU
a_jax = jnp.array(a_np)
b_jax = jnp.array(b_np)
C_tpu = jnp.matmul(a_jax, b_jax).block_until_ready()
C_tpu_np = np.array(C_tpu)  # Vraćamo na CPU za poređenje

max_err = np.max(np.abs(C_ref - C_tpu_np))
rel_err = np.mean(np.abs(C_ref - C_tpu_np) / (np.abs(C_ref) + 1e-8))

print(f"Maksimalna apsolutna greška: {max_err:.6f}")
print(f"Prosečna relativna greška:   {rel_err:.2e}")
print(f"Tačnost: {'✓ OK (< 1e-4)' if max_err < 1e-4 else '✗ Prevelika greška!'}")

# ── 8. Finalni rezime ─────────────────────────────────────────
print("\n" + "="*60)
print("REZIME REZULTATA")
print("="*60)
print(f"Matrica {N}×{N}:")
print(f"  float32 (jit):  {avg_time:.3f} ms")
print(f"  bfloat16 (jit): {avg_bf16:.3f} ms")
if num_devices > 1:
    print(f"  bfloat16 (pmap, {num_devices} čipova, batch={num_devices}): {avg_pmap:.3f} ms")
print("\nNapomena: Warm-up (JIT kompajliranje) se ne računa u benchmark.")
print("Koristiti .block_until_ready() uvek pri merenju TPU operacija.")