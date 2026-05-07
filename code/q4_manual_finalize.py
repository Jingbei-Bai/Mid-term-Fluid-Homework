from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

OUTDIR = Path('/mnt/data/problem4_out')
OUTDIR.mkdir(exist_ok=True)

# Build resolution histories from already computed values
rows = [
    # DRP
    ('DRP',64,0.1291710683552999),('DRP',128,0.09918277750247524),('DRP',256,0.06731727812883767),('DRP',512,0.011411772910115605),('DRP',960,0.0011111457544979648),('DRP',992,0.0009790762088493611),
    # DRP-M
    ('DRP-M',64,0.1330642152013083),('DRP-M',128,0.10688495208622546),('DRP-M',256,0.019829642731722136),('DRP-M',512,0.004008714688515286),('DRP-M',768,0.001023785090584431),('DRP-M',800,0.0008825271710052764),
    # MDCD
    ('MDCD',64,0.08557623950546729),('MDCD',128,0.07741717647436151),('MDCD',256,0.0333942314258463),('MDCD',512,0.007078510337453866),('MDCD',864,0.001078784962535862),('MDCD',896,0.0009392410031057649),
    # SA-DRP
    ('SA-DRP',64,0.09553348126443395),('SA-DRP',128,0.10275361932050564),('SA-DRP',256,0.043219090104524335),('SA-DRP',512,0.0016659956342765063),('SA-DRP',544,0.0011765058257604745),('SA-DRP',576,0.0008423672995603055),
    # UW1
    ('UW1',512,0.08905727112808251),('UW1',1024,0.08855661140146723),('UW1',1536,0.08751861718849778),('UW1',2048,0.08612184946527723),('UW1',3072,0.08287872060639827),('UW1',4096,0.07950353983634291),
    # UW2
    ('UW2',2048,0.04890048245906148),('UW2',3072,0.024468445360670738),('UW2',4096,0.014134982167277113),
    # UW3
    ('UW3',512,0.04263742533057558),('UW3',1024,0.011577552119818415),('UW3',2048,0.001742020141387381),('UW3',2304,0.001234692270810293),('UW3',2560,0.000905601614266422),
]

hist = pd.DataFrame(rows, columns=['scheme','N','L1_error'])
hist.to_csv(OUTDIR/'q4_resolution_histories.csv', index=False)

resolution = pd.DataFrame([
    {'scheme':'UW1','min_N_for_L1<=1e-3':'>4096'},
    {'scheme':'UW2','min_N_for_L1<=1e-3':'>4096'},
    {'scheme':'UW3','min_N_for_L1<=1e-3':2560},
    {'scheme':'DRP','min_N_for_L1<=1e-3':992},
    {'scheme':'DRP-M','min_N_for_L1<=1e-3':800},
    {'scheme':'MDCD','min_N_for_L1<=1e-3':896},
    {'scheme':'SA-DRP','min_N_for_L1<=1e-3':576},
])
resolution.to_csv(OUTDIR/'q4_resolution_threshold.csv', index=False)

plt.figure(figsize=(8.8,5.8))
for s in ['UW1','UW2','UW3','DRP','DRP-M','MDCD','SA-DRP']:
    sub = hist[hist.scheme==s].sort_values('N')
    plt.semilogy(sub['N'], sub['L1_error'], marker='o', markersize=4, linewidth=1.4, label=s)
plt.axhline(1e-3, color='k', linestyle='--', linewidth=1.2, label='1e-3 threshold')
plt.xlabel('N')
plt.ylabel(r'$L_1$ error')
plt.title('Q4 resolution study')
plt.grid(True, which='both', alpha=0.3)
plt.legend(frameon=False, ncol=2)
plt.tight_layout()
plt.savefig(OUTDIR/'q4_resolution.png', dpi=220)
plt.close()

conv = pd.read_csv(OUTDIR/'q4_convergence.csv')
summary = []
summary.append('Q4 numerical experiment summary\n')
summary.append('Seed = 20260420\n')
summary.append('Equation: u_t + u_x = 0, x in [0,1], periodic BC.\n')
summary.append('Initial condition used in the code/report:\n')
summary.append('u(x,0) = 1 + epsilon * sum_{k=1}^{64} sqrt(E(k)) * sin(2*pi*k*(x+psi_k)),\n')
summary.append('E(k) = (k/k0)^4 exp(-2 (k/k0)^2), epsilon = 0.1, k0 = 24, psi_k ~ U[0,1].\n')
summary.append('The random phases are frozen using a fixed seed for reproducibility. CFL = 0.2, t_end = 1.0.\n\n')
summary.append('Convergence table:\n')
summary.append(conv.to_string(index=False))
summary.append('\n\nResolution threshold table:\n')
summary.append(resolution.to_string(index=False))
(OUTDIR/'q4_summary.txt').write_text(''.join(summary), encoding='utf-8')
