import numpy as np

scale = 1.02
base_size       = 4.0   * scale
plate_thickness = 0.2   * scale
r_base          = 0.09  * scale
r_tip           = 0.001 * scale
needle_height   = 0.50  * scale
gap             = 0.01  * scale
spacing         = 2*r_base + gap
fn              = 64

count  = int(np.floor(base_size/spacing))
offset = (base_size - (count-1)*spacing)/2.0

tris = []

def add_box(t, mn, mx):
    x0,y0,z0 = mn; x1,y1,z1 = mx
    v = np.array([
        [x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],
        [x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]
    ])
    faces = [(0,1,2,3),(4,5,6,7),(0,1,5,4),
             (1,2,6,5),(2,3,7,6),(3,0,4,7)]
    for i,(a,b,c,d) in enumerate(faces):
        if i==0:  # bottom face: reverse winding
            t.append((v[a],v[c],v[b]))
            t.append((v[a],v[d],v[c]))
        else:
            t.append((v[a],v[b],v[c]))
            t.append((v[a],v[c],v[d]))

def add_frustum(t, tx, ty):
    z0, z1 = plate_thickness, plate_thickness + needle_height
    angs = np.linspace(0,2*np.pi,fn,endpoint=False)
    base_ring = [(tx + r_base*np.cos(a), ty + r_base*np.sin(a), z0) for a in angs]
    if r_tip < 1e-3:
        tip = (tx, ty, z1)
        for i in range(fn):
            j = (i+1)%fn
            t.append((base_ring[i], base_ring[j], tip))
    else:
        top_ring = [(tx + r_tip*np.cos(a), ty + r_tip*np.sin(a), z1) for a in angs]
        for i in range(fn):
            j = (i+1)%fn
            b0,b1 = base_ring[i], base_ring[j]
            t0,t1 = top_ring[i],   top_ring[j]
            t.append((b0,b1,t1)); t.append((b0,t1,t0))
        center = (tx, ty, z1)
        for i in range(fn):
            j = (i+1)%fn
            t.append((center, top_ring[j], top_ring[i]))

add_box(tris, (0,0,0), (base_size, base_size, plate_thickness))
for i in range(count):
    for j in range(count):
        x = offset + i*spacing
        y = offset + j*spacing
        add_frustum(tris, x, y)

with open('micro_array_sharp_scaled.stl','w') as f:
    f.write('solid array\n')
    for p0,p1,p2 in tris:
        u = np.subtract(p1,p0); v = np.subtract(p2,p0)
        n = np.cross(u,v)
        if np.linalg.norm(n)>0: n /= np.linalg.norm(n)
        f.write(f'  facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}\n')
        f.write('    outer loop\n')
        for p in (p0,p1,p2):
            f.write(f'      vertex {p[0]:.6e} {p[1]:.6e} {p[2]:.6e}\n')
        f.write('    endloop\n  endfacet\n')
    f.write('endsolid array\n')
