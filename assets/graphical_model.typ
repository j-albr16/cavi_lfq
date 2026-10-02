// The graphical model of lfq.typ as a standalone image (used by README.md).
// Build: typst compile assets/graphical_model.typ assets/graphical_model.png --ppi 220
#import "../style.typ": gm

#set page(width: auto, height: auto, margin: 8pt, fill: white)

#gm(
  (
    (id: "sigma", p: (0.6, 0.5), label: $sigma_k^2$),
    (id: "mu", p: (0.6, 2.1), label: $mu_k$),
    (id: "t", p: (2.0, 3.8), label: $t_n$, obs: true),
    (id: "pi", p: (3.4, 0.5), label: $pi$),
    (id: "z", p: (3.4, 2.1), label: $z_n$),
    (id: "n", p: (4.8, 0.5), label: $n_k$),
    (id: "p", p: (6.2, 0.5), label: $p$),
    (id: "j", p: (5.4, 2.1), label: $j_n$),
    (id: "y", p: (5.9, 3.8), label: $y_n$, obs: true),
    (id: "s", p: (7.4, 2.1), label: $s^2$),
    (id: "m", p: (8.7, 2.1), label: $m_k$),
    (id: "c", p: (8.7, 3.8), label: $c_k$),
  ),
  (
    ("sigma", "mu"), ("sigma", "t"), ("mu", "t"), ("z", "t"),
    ("pi", "z"), ("z", "j"), ("n", "j"), ("p", "j"),
    ("z", "y"), ("j", "y"), ("s", "y"), ("m", "y"), ("c", "y"),
  ),
  plates: ((x0: 1.3, y0: 1.5, x1: 6.8, y1: 4.5, label: $N$),),
)
