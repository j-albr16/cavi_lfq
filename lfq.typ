// ============================================================
//  Variational Bayes for LC–MS feature detection
// ============================================================
#import "style.typ": *
#show: template

// ============================================================
//  Title
// ============================================================

#align(center)[
  #v(0.5em)
  #text(size: 19pt, weight: "bold", fill: accent)[Variational Bayes for LC–MS Feature Detection]
  #v(0.2em)
  #text(size: 11pt, fill: luma(80))[A mean-field model for elution profiles, isotope patterns and charge states]
  #v(0.8em)
]

#block(inset: (x: 1.2cm))[
  #set text(size: 9.5pt)
  #set par(justify: true)
  *Summary.* An LC–MS map is treated as a weighted point cloud in (retention time, m/z). Each pixel is generated
  either by one of $K$ features or by a uniform background. A feature has a Gaussian elution profile in RT and an
  isotope comb in m/z, whose peak heights follow a binomial in the carbon count $n_k$ and the #super[13]C probability $p$.
  RT and m/z are conditionally independent given the feature. We derive coordinate-ascent variational inference (CAVI)
  updates for all factors of a structured mean-field approximation.
]

#outline(indent: auto, depth: 2)

// ============================================================
= Model
// ============================================================

== Data

The map is a grid of non-empty pixels $n = 1, dots, N$. Each pixel has a retention time $t_n$, an m/z value $y_n$
and an intensity, converted to an *effective ion count* $w_n = gamma I_n$. All $w_n$ ions of a pixel share the same
$(t_n, y_n)$, so every sum over ions becomes a weighted sum over pixels. The constant
$Delta = 1.00336$ Da is the mass difference between #super[13]C and #super[12]C.

== Generative story

Every ion is produced as follows:

+ *Which feature?* $z_n tilde "Cat"(pi)$, where $z_n = 0$ means background.
+ *Background* ($z_n = 0$): $(t_n, y_n)$ is uniform over the window, $p(t_n, y_n | z_n = 0) = 1 slash (T Y)$.
+ *Feature* $k >= 1$:
  - *How many #super[13]C atoms?* $j_n | z_n = k tilde cal(B)(n_k, p)$
  - *Retention time:* $t_n | z_n = k tilde cal(N)(mu_k, sigma_k^2)$
  - *m/z:* $y_n | z_n = k, j_n tilde cal(N)(m_k + j_n Delta slash c_k, s^2)$

The two main assumptions are that the isotope pattern is independent of RT *given the feature* (all isotopologues
co-elute), and that the m/z peaks of all features share one global width $s^2$.

#result(title: "Update: responsibilities")[
  We can setup the generative model:

  $
    p(t_i,y_i|,z_i,j_i,theta) = cases(pi_0/(T Y) & k=0 "(background)",cal(N)(t_i;mu_k,sigma_k^2)cal(N)(y_i;m_k + (j_i Delta)/c_k,s^2) & k>= 1)
  $
]



== Graphical model

#figure(
  gm(
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
  ),
  caption: [Graphical model. Shaded nodes are observed. Variables with index $k$ exist once per feature
    (plate over $K$ omitted); $p$ and $s^2$ are global. Hyperparameters are not shown.],
)

== Variables

#figure(
  btable(
    columns: (auto, 1fr, auto, auto),
    header: ([Symbol], [Meaning], [Scope], [Type]),
    $z_n$, [feature assignment of pixel $n$ ($z_n = 0$: background)], [pixel], [latent, categorical],
    $j_n$, [isotope index: number of #super[13]C atoms of the ion], [pixel], [latent, discrete],
    $t_n$, [retention time], [pixel], [observed],
    $y_n$, [m/z], [pixel], [observed],
    $w_n$, [effective ion count (weight)], [pixel], [observed],
    $pi_k$, [relative abundance of feature $k$ ($pi_0$: background)], [global], [latent],
    $mu_k$, [apex retention time], [feature], [latent],
    $sigma_k^2$, [RT variance (elution width)], [feature], [latent],
    $m_k$, [monoisotopic m/z], [feature], [latent],
    $c_k$, [charge, $c_k in {1, 2, 3, 4}$], [feature], [latent, discrete],
    $n_k$, [number of carbon atoms], [feature], [latent, discrete],
    $p$, [#super[13]C probability], [global], [latent],
    $s^2$, [m/z peak variance], [global], [latent],
    $Delta$, [#super[13]C – #super[12]C mass difference], [—], [constant],
  ),
  caption: [Variables of the model.],
)

// ============================================================
= Priors
// ============================================================

All priors are (conditionally) conjugate, except for the carbon count, which lives on a finite grid.

#figure(
  btable(
    columns: (auto, auto, 1fr),
    header: ([Variable], [Prior], [Reason]),
    $pi$, $"Dir"(alpha)$, [conjugate to the categorical $z$; $alpha_k < 1$ prunes unused components],
    $(mu_k, sigma_k^2)$, $"NIG"(m_(0 k), nu_(0 k), a_0, b_0)$, [conjugate for a Gaussian with unknown mean and variance],
    $m_k$, $cal(N)(hat(m)_k, tau_k^2)$, [conjugate given $c_k$, $j$ and $s^2$],
    $c_k$, [$"Cat"(rho)$ on ${1,2,3,4}$], [small discrete support, enumerated exactly],
    $n_k$, $p(n_k | hat(m)_k)$, [no conjugate prior for the binomial's $n$; exact on a grid],
    $p$, $"Beta"(a_p, b_p)$, [conjugate to the binomial in $p$],
    $s^2$, $"IG"(a_s, b_s)$, [conjugate for the variance of a Gaussian with known mean],
  ),
  caption: [Priors at a glance.],
)

The properties below are stated for generic parameters. They are used with the *variational* parameters
(marked with a tilde) in the updates.

=== Mixing weights: $pi tilde "Dir"(alpha)$

The Dirichlet runs over $K + 1$ components including the background. Choosing $alpha_k < 1$ lets VB push unused
components towards zero weight, so one can start with a few more candidates than necessary. A larger $alpha_0$
gives the background more prior weight if a lot of noise is expected.

$ log p(pi) = sum_(k=0)^K (alpha_k - 1) log pi_k + "const" $
$ EE[log pi_k] = psi(alpha_k) - psi(sum_(k'=0)^K alpha_(k')), quad EE[pi_k] = alpha_k / (sum_(k') alpha_(k')) $

=== RT profile: $(mu_k, sigma_k^2) tilde "NIG"(m_(0 k), nu_(0 k), a_0, b_0)$

This means $mu_k | sigma_k^2 tilde cal(N)(m_(0 k), sigma_k^2 slash nu_(0 k))$ and $sigma_k^2 tilde "IG"(a_0, b_0)$.
$m_(0 k)$ is the candidate's RT, $nu_(0 k)$ says how much we trust it, and $a_0$, $b_0$ encode the typical
chromatographic peak width.

$
  log p(mu_k, sigma_k^2)
  &= -1/2 log sigma_k^2 - (nu_(0 k) (mu_k - m_(0 k))^2) / (2 sigma_k^2) - (a_0 + 1) log sigma_k^2 - b_0 / sigma_k^2 + "const" \
  &= -(a_0 + 3/2) log sigma_k^2 - 1/sigma_k^2 ((nu_(0 k) (mu_k - m_(0 k))^2) / 2 + b_0) + "const"
$

For $"NIG"(m, nu, a, b)$:
$ EE[1 / sigma^2] = a / b, quad EE[log sigma^2] = log b - psi(a), quad EE[mu | sigma^2] = m, quad "Var"[mu | sigma^2] = sigma^2 / nu $

=== Monoisotopic m/z: $m_k tilde cal(N)(hat(m)_k, tau_k^2)$

Given $c_k$, $j$ and $s^2$, the m/z likelihood is Gaussian in $m_k$, so this prior is conjugate. $hat(m)_k$ is the
candidate from the peak picker, and $tau_k$ corresponds to a few ppm of $hat(m)_k$.

$ log p(m_k) = - (m_k - hat(m)_k)^2 / (2 tau_k^2) + "const" $

=== Charge: $c_k tilde "Cat"(rho)$ on ${1, 2, 3, 4}$

The support is small and discrete, so it is enumerated exactly. $rho$ reflects typical charge distributions; for
tryptic peptides, 2+ dominates. The prior is shared, the posterior $q(c_k)$ is individual per feature.

$ log p(c_k = c) = log rho_c, quad EE[f(c_k)] = sum_c q(c_k = c) f(c) $

=== Carbon count: $n_k tilde p(n_k | hat(m)_k)$

An empirical categorical from an in-silico digest, evaluated at the candidate mass $hat(m)_k$. There is no conjugate
prior for the number of trials of a binomial, but the grid is exact and cheap. Conditioning on the candidate mass
(not on the latent $m_k$) keeps $q(m_k)$ Gaussian. Expectations are finite sums:

$ EE[n_k] = sum_C q(n_k = C) C, quad EE[log binom(n_k, j)] = sum_C q(n_k = C) log binom(C, j) $

=== Heavy-isotope probability: $p tilde "Beta"(a_p, b_p)$

The mean $a_p slash (a_p + b_p) approx 0.0107$ is the natural #super[13]C abundance, and a large $a_p + b_p$ makes the
prior tight. This also resolves the $n$–$p$ trade-off.

$ log p(p) = (a_p - 1) log p + (b_p - 1) log(1 - p) + "const" $
$ EE[log p] = psi(a_p) - psi(a_p + b_p), quad EE[log(1 - p)] = psi(b_p) - psi(a_p + b_p) $

=== m/z peak width: $s^2 tilde "IG"(a_s, b_s)$

The prior mean comes from the instrument's resolution or mass accuracy.

$ log p(s^2) = -(a_s + 1) log s^2 - b_s / s^2 + "const" $
$ EE[1 / s^2] = a_s / b_s, quad EE[log s^2] = log b_s - psi(a_s) $

// ============================================================
= Variational approximation
// ============================================================

== Mean-field family

We approximate the posterior with the structured mean-field family

$ q = q(z, j) thin q(pi) thin q(mu, sigma^2) thin q(m, c) thin q(n) thin q(p) thin q(s^2). $

$z$ and $j$ are kept jointly, because the isotope index only has a meaning relative to a feature. $m$ and $c$ are
kept jointly, because the charge sets the comb spacing and hence the best mass. The factorizations over pixels $n$
and features $k$ are not assumptions: they follow from the model (induced factorizations).

== Coordinate ascent

Since
$ q^* = arg min_(q in cal(Q)) D_"KL" (q(theta) || p(theta | x)) = arg max_(q in cal(Q)) cal(L)(q), $
the optimal factor $q_i$, with all other factors fixed, is
$ log q_i^*(theta_i) = EE_(q_(-i)) [log p(x, theta)] + "const", $
where the expectation runs over all factors except $q_i$. Each update maximizes the ELBO with respect to one
factor, so the ELBO increases monotonically.

Because:

$
  cal(L)(q_i) &= EE_(q, j)[log p(x,z_j) / q(z_j)] \
  &= EE_(q, j)[log p(x,z_j)] + sum_i HH[q_i] \
  &= EE_q_i [EE_(q, j!=i)[log p(x,z_j)]] + HH[q_i] + sum_(i!=j) HH[q_j]  \
  &= EE_q_i [EE_(q, j!=i)[log p(x,z_j)]] + HH[q_i] + sum_(i!=j) HH[q_j]  \
  &= EE_q_i [tilde(p)(x, z_i)] + HH[q_i] + sum_(i!=j) HH[q_j]  \
  &= D_"KL" (q_i || tilde(p)(x,z_i)) + "const"
$

== Log joint

$
  log p = & underbrace(log p(z | pi) + log p(j | z, n, p) + log p(t | z, mu, sigma^2) + log p(y | z, j, m, s^2, c), "likelihood terms") \
  &quad + med underbrace(log p(pi) + log p(mu, sigma^2) + log p(m) + log p(c) + log p(n) + log p(p) + log p(s^2), "priors")
$

Each update only needs the terms that contain its variable (its Markov blanket); everything else goes into the
constant.

== Notation and identities

#figure(
  btable(
    columns: (auto, auto, 1fr),
    header: ([Symbol], [Definition], [Meaning]),
    $r_(n k j)$, $EE[z_(n k) bb(1)[j_n = j]]$, [responsibility of sub-peak $(k, j)$ for pixel $n$],
    $r_(n 0)$, $EE[z_(n 0)]$, [responsibility of the background],
    $R_(n k)$, $sum_j r_(n k j)$, [responsibility of feature $k$],
    $N_k$, $sum_n w_n R_(n k)$, [expected number of ions of feature $k$],
    $lambda$, $EE[1 slash s^2] = tilde(a)_s slash tilde(b)_s$, [expected m/z precision],
    $u_(n j)(c)$, $y_n - j Delta slash c$, [monoisotopic m/z implied by pixel $n$ as isotope $j$ at charge $c$],
    $S$, $sum_n sum_k sum_j w_n r_(n k j) j$, [expected total number of #super[13]C atoms],
  ),
  caption: [Shorthand used in the updates.],
)

The following identities are used repeatedly:

/ Expected square: $EE[(X - a)^2] = "Var"[X] + (EE[X] - a)^2$ for any random variable $X$ and constant $a$.
/ Tower rule: $EE_(q(x, v))[f] = EE_(q(v))[EE_(q(x | v))[f]]$.
/ Assignments: $EE_(q(z, j))[z_(n k) f(j_n)] = sum_j r_(n k j) f(j)$.
/ Completing the square: with weights $omega_i$ and centers $u_i$, and $A = sum_i omega_i$,
  $B = sum_i omega_i u_i$, $C = sum_i omega_i u_i^2$:
  $ sum_i omega_i (x - u_i)^2 = A (x - B/A)^2 + (C - B^2 / A), quad C - B^2 / A = sum_i omega_i (u_i - B/A)^2. $
/ Gaussian integral: $integral exp(-A/2 (x - x_0)^2) dif x = sqrt(2 pi slash A)$.

#remark(title: "Numerics")[
  $C - B^2 slash A$ is the difference of two huge, nearly equal numbers (squared m/z values around $10^5$ times a
  large precision). Always evaluate it in the deviation form $sum_i omega_i (u_i - B slash A)^2$, which only
  sums small positive terms. Normalize all discrete distributions with log-sum-exp.
]

// ============================================================
= Updates
// ============================================================

== Assignments: $q(z, j)$

Collecting the four likelihood terms and taking expectations over all other factors:

$
  log q^*(z, j)
  &= EE_(q(pi))[log p(z | pi)] + EE_(q(n), q(p))[log p(j | z, n, p)] \
  &quad + med EE_(q(mu, sigma^2))[log p(t | z, mu, sigma^2)] + EE_(q(m, c), q(s^2))[log p(y | z, j, m, s^2, c)] + "const" \
  &= sum_n (z_(n 0) log rho_(n 0) + sum_(k=1)^K sum_j z_(n k) bb(1)[j_n = j] log rho_(n k j)) + "const",
$

with one term per sub-peak $(k, j)$:

$
  log rho_(n k j) = & EE[log pi_k] + EE[log cal(B)(j | n_k, p)] \
  &quad + med EE[log cal(N)(t_n | mu_k, sigma_k^2)] + EE[log cal(N)(y_n | m_k + j Delta slash c_k, s^2)].
$

We evaluate the four expectations one by one.

*Mixing weight.* From the Dirichlet:
$ EE[log pi_k] = psi(tilde(alpha)_k) - psi(sum_(k'=0)^K tilde(alpha)_(k')). $

*Isotope pattern.* Expanding the binomial, and using that $n_k$ and $p$ are independent under $q$:
$
  EE[log cal(B)(j | n_k, p)]
  &= EE[log binom(n_k, j)] + j thin EE[log p] + (EE[n_k] - j) EE[log(1 - p)] \
  &= sum_C q(n_k = C) log binom(C, j) + j (psi(tilde(a)_p) - psi(tilde(b)_p)) + EE[n_k] (psi(tilde(b)_p) - psi(tilde(a)_p + tilde(b)_p)).
$

*Retention time.* $mu_k$ and $sigma_k^2$ are dependent under $q$, so we average over $mu_k | sigma_k^2$ first:
$
  EE[log cal(N)(t_n | mu_k, sigma_k^2)]
  &= -1/2 (log 2 pi + EE[log sigma_k^2] + EE_(q(sigma_k^2))[1/sigma_k^2 EE_(q(mu_k | sigma_k^2))[(t_n - mu_k)^2]]) \
  &= -1/2 (log 2 pi + EE[log sigma_k^2] + EE_(q(sigma_k^2))[1/sigma_k^2 (sigma_k^2 / tilde(nu)_k + (t_n - tilde(mu)_k)^2)]) \
  &= -1/2 (log 2 pi + log tilde(b)_k - psi(tilde(a)_k) + 1/tilde(nu)_k + tilde(a)_k / tilde(b)_k (t_n - tilde(mu)_k)^2).
$

*m/z.* $s^2$ is independent of $(m_k, c_k)$. For the squared residual we use the tower rule over the charge and
the expected square; the conditional variance $tilde(tau)_k^2$ does not depend on $c$ (see @sec-mc):
$
  EE[log cal(N)(y_n | m_k + j Delta slash c_k, s^2)]
  &= -1/2 (log 2 pi + EE[log s^2] + EE[1/s^2] thin EE_(q(m_k, c_k))[(m_k + (j Delta) / c_k - y_n)^2]) \
  &= -1/2 (log 2 pi + log tilde(b)_s - psi(tilde(a)_s) + lambda (tilde(tau)_k^2 + sum_c q(c_k = c) (tilde(m)_(k c) + (j Delta) / c - y_n)^2)).
$

*Background.* No RT, m/z or isotope structure:
$ log rho_(n 0) = psi(tilde(alpha)_0) - psi(sum_(k'=0)^K tilde(alpha)_(k')) - log(T Y). $

#remark(title: "Constants")[
  The $log 2 pi$ terms are the same for every feature, but they differ from the background's $log(T Y)$.
  They must therefore be kept when the background is part of the normalization.
]

Exponentiating gives a categorical distribution over the sub-peaks of each pixel plus the background:
$ q^*(z, j) prop product_n (rho_(n 0)^(z_(n 0)) product_(k=1)^K product_j rho_(n k j)^(z_(n k) bb(1)[j_n = j])). $

#result(title: "Update: responsibilities")[
  $
    r_(n k j) = rho_(n k j) / (rho_(n 0) + sum_(k'=1)^K sum_(j') rho_(n k' j')),
    quad r_(n 0) = rho_(n 0) / (rho_(n 0) + sum_(k'=1)^K sum_(j') rho_(n k' j')),
    quad R_(n k) = sum_j r_(n k j)
  $
  Computed in log space with log-sum-exp. The pixel weights $w_n$ do not enter here.
]

== Mixing weights: $q(pi)$

Only $log p(z | pi)$ and the prior contain $pi$. With $R_(n 0) := r_(n 0)$:

$
  log q^*(pi)
  &= EE_(q(z))[log p(z | pi)] + log p(pi) + "const" \
  &= sum_n sum_(k=0)^K w_n R_(n k) log pi_k + sum_(k=0)^K (alpha_k - 1) log pi_k + "const" \
  &= sum_(k=0)^K (alpha_k + N_k - 1) log pi_k + "const".
$

Matching with $log "Dir"(pi | tilde(alpha)) = sum_k (tilde(alpha)_k - 1) log pi_k + "const"$:

#result(title: "Update: Dirichlet")[
  $ q^*(pi) = "Dir"(tilde(alpha)), quad tilde(alpha)_k = alpha_k + N_k, quad k = 0, dots, K $
]

== RT profiles: $q(mu, sigma^2)$

The expression is a sum over features, so $q^*(mu, sigma^2) = product_k q^*(mu_k, sigma_k^2)$. For one feature:

$
  log q^*(mu_k, sigma_k^2)
  &= sum_n w_n R_(n k) (-1/2 log sigma_k^2 - 1/(2 sigma_k^2) (t_n - mu_k)^2) + log p(mu_k, sigma_k^2) + "const" \
  &= -(N_k / 2 + 1/2 + a_0 + 1) log sigma_k^2 \
  &quad - med 1/sigma_k^2 (b_0 + 1/2 [nu_(0 k) (mu_k - m_(0 k))^2 + sum_n w_n R_(n k) (t_n - mu_k)^2]) + "const".
$

*Completing the square.* The bracket is a weighted sum of squares with centers $m_(0 k)$ (weight $nu_(0 k)$) and
$t_n$ (weights $w_n R_(n k)$):

$
  & nu_(0 k) (mu_k - m_(0 k))^2 + sum_n w_n R_(n k) (t_n - mu_k)^2 \
  &= underbrace((nu_(0 k) + N_k), A) mu_k^2 - 2 underbrace((nu_(0 k) m_(0 k) + sum_n w_n R_(n k) t_n), B) mu_k + underbrace((nu_(0 k) m_(0 k)^2 + sum_n w_n R_(n k) t_n^2), C) \
  &= A (mu_k - B / A)^2 + (C - B^2 / A).
$

*Splitting into Gaussian and inverse-gamma.* The Gaussian over $mu_k | sigma_k^2$ needs its own
$-1/2 log sigma_k^2$ for normalization; the rest belongs to the inverse-gamma:

$
  log q^*(mu_k, sigma_k^2)
  = underbrace(-1/2 log sigma_k^2 - A / (2 sigma_k^2) (mu_k - B / A)^2, log cal(N)(mu_k | B slash A, thin sigma_k^2 slash A))
  underbrace(- (a_0 + N_k / 2 + 1) log sigma_k^2 - 1/sigma_k^2 (b_0 + 1/2 (C - B^2 / A)), log "IG"(sigma_k^2))
  + "const".
$

#result(title: "Update: Normal-inverse-gamma")[
  $ q^*(mu_k, sigma_k^2) = "NIG"(tilde(mu)_k, tilde(nu)_k, tilde(a)_k, tilde(b)_k) $
  $
    tilde(nu)_k &= nu_(0 k) + N_k,
    & tilde(mu)_k &= (nu_(0 k) m_(0 k) + sum_n w_n R_(n k) t_n) / (nu_(0 k) + N_k), \
    tilde(a)_k &= a_0 + N_k / 2,
    & tilde(b)_k &= b_0 + 1/2 (nu_(0 k) (m_(0 k) - tilde(mu)_k)^2 + sum_n w_n R_(n k) (t_n - tilde(mu)_k)^2).
  $
  The last form of $tilde(b)_k$ equals $b_0 + 1/2 (C - B^2 slash A)$ and is numerically stable.
]

== Mass and charge: $q(m, c)$ <sec-mc>

The terms with $log s^2$ do not depend on $(m, c)$ and go into the constant. With $lambda = EE[1 slash s^2]$:

$
  log q^*(m, c)
  &= EE_(q(z, j), q(s^2))[log p(y | z, j, m, s^2, c)] + log p(m) + log p(c) + "const" \
  &= - sum_n sum_k EE[1/(2 s^2)] thin EE_(q(z, j))[z_(n k) (m_k + (j_n Delta) / c_k - y_n)^2] + log p(m) + log p(c) + "const" \
  &= sum_k (-lambda / 2 sum_n sum_j w_n r_(n k j) (m_k + (j Delta) / c_k - y_n)^2 - (m_k - hat(m)_k)^2 / (2 tau_k^2) + log rho_(c_k)) + "const".
$

It factorizes over features. For one feature and one *fixed charge hypothesis* $c_k = c$, rewrite the residual with
the implied monoisotopic position $u_(n j)(c) = y_n - j Delta slash c$:

$
  & lambda sum_n sum_j w_n r_(n k j) (m_k - u_(n j)(c))^2 + 1 / tau_k^2 (m_k - hat(m)_k)^2 \
  &= underbrace((lambda N_k + 1 / tau_k^2), A) m_k^2 - 2 underbrace((lambda sum_n sum_j w_n r_(n k j) u_(n j)(c) + hat(m)_k / tau_k^2), B_c) m_k + underbrace((lambda sum_n sum_j w_n r_(n k j) u_(n j)(c)^2 + hat(m)_k^2 / tau_k^2), C_c) \
  &= A (m_k - B_c / A)^2 + (C_c - B_c^2 / A).
$

The weights do not depend on $c$, only the centers do; hence $A$ is the same for every charge. Putting the
$-1/2$ and the charge prior back:

$ log q^*(m_k, c_k = c) = -A / 2 (m_k - B_c / A)^2 - 1/2 (C_c - B_c^2 / A) + log rho_c + "const". $

*Conditional on the charge*, this is a Gaussian in $m_k$ with precision $A$ and mean $B_c slash A$.

*Marginal over the mass.* Integrating $m_k$ out with the Gaussian integral:

$
  q^*(c_k = c)
  &= integral q^*(m_k, c_k = c) dif m_k
  prop rho_c exp(-1/2 (C_c - B_c^2 / A)) sqrt((2 pi) / A) \
  ==> log q^*(c_k = c) &= log rho_c - 1/2 (C_c - B_c^2 / A) + "const",
$

since $A$ does not depend on $c$.

#result(title: "Update: mass and charge")[
  $ q^*(m_k, c_k) = q^*(c_k) thin cal(N)(m_k | tilde(m)_(k c_k), tilde(tau)_k^2) $
  $
    tilde(tau)_k^2 = 1 / (lambda N_k + 1 slash tau_k^2),
    quad tilde(m)_(k c) = tilde(tau)_k^2 (lambda sum_n sum_j w_n r_(n k j) (y_n - (j Delta) / c) + hat(m)_k / tau_k^2)
  $
  $
    q^*(c_k = c) = exp(ell_c) / (sum_(c') exp(ell_(c'))),
    quad ell_c = log rho_c - 1/2 (lambda sum_n sum_j w_n r_(n k j) (y_n - (j Delta) / c - tilde(m)_(k c))^2 + (hat(m)_k - tilde(m)_(k c))^2 / tau_k^2)
  $
  $ell_c$ is written in the stable deviation form of $C_c - B_c^2 slash A$. It measures how well the comb with
  spacing $Delta slash c$ lines up with the data.
]

== m/z peak width: $q(s^2)$

$s^2$ is global, so the sums run over all features $k >= 1$. The inner expectation is the same as in the $q(z, j)$
update:

$
  log q^*(s^2)
  &= EE_(q(z, j), q(m, c))[log p(y | z, j, m, s^2, c)] + log p(s^2) + "const" \
  &= - sum_n sum_k w_n R_(n k) 1/2 log s^2 - 1/(2 s^2) sum_n sum_k sum_j w_n r_(n k j) EE_(q(m_k, c_k))[(m_k + (j Delta) / c_k - y_n)^2] \
  &quad - med (a_s + 1) log s^2 - b_s / s^2 + "const" \
  &= -(a_s + 1 + 1/2 sum_(k=1)^K N_k) log s^2 \
  &quad - med 1/s^2 (b_s + 1/2 sum_n sum_k sum_j w_n r_(n k j) (tilde(tau)_k^2 + sum_c q(c_k = c) (tilde(m)_(k c) + (j Delta) / c - y_n)^2)) + "const".
$

#result(title: "Update: inverse-gamma")[
  $
    tilde(a)_s &= a_s + 1/2 sum_(k=1)^K N_k, \
    tilde(b)_s &= b_s + 1/2 sum_n sum_k sum_j w_n r_(n k j) (tilde(tau)_k^2 + sum_c q(c_k = c) (tilde(m)_(k c) + (j Delta) / c - y_n)^2),
    quad lambda = tilde(a)_s / tilde(b)_s.
  $
]

== Carbon counts: $q(n)$

The term $j log p$ does not contain $n_k$ and goes into the constant; so does $-j log(1 - p)$. The prior is a
separate table per feature, $log p(n) = sum_k log p(n_k | hat(m)_k)$:

$
  log q^*(n)
  &= EE_(q(z, j), q(p))[log p(j | z, n, p)] + log p(n) + "const" \
  &= sum_n sum_k sum_j w_n r_(n k j) (log binom(n_k, j) + j thin EE[log p] + (n_k - j) EE[log(1 - p)]) + log p(n) + "const" \
  &= sum_k (sum_n sum_j w_n r_(n k j) log binom(n_k, j) + n_k N_k thin EE[log(1 - p)] + log p(n_k | hat(m)_k)) + "const".
$

There is no parametric family to recognize: $q(n_k)$ is evaluated on the grid.

#result(title: "Update: carbon count (grid)")[
  $
    ell_k (C) = log p(n_k = C | hat(m)_k) + sum_n sum_j w_n r_(n k j) log binom(C, j) + C N_k (psi(tilde(b)_p) - psi(tilde(a)_p + tilde(b)_p))
  $
  $ q^*(n_k = C) = exp(ell_k (C)) / (sum_(C') exp(ell_k (C'))) $
  The grid starts at the largest allowed isotope index, since $binom(C, j) = 0$ for $C < j$.
]

== Heavy-isotope probability: $q(p)$

The binomial coefficient does not contain $p$ and goes into the constant. $EE[n_k]$ does not depend on $n$ or $j$,
and $sum_n sum_j w_n r_(n k j) = N_k$:

$
  log q^*(p)
  &= EE_(q(z, j), q(n))[log p(j | z, n, p)] + log p(p) + "const" \
  &= sum_n sum_k sum_j w_n r_(n k j) (j log p + (EE[n_k] - j) log(1 - p)) \
  &quad + med (a_p - 1) log p + (b_p - 1) log(1 - p) + "const" \
  &= (a_p + S - 1) log p + (b_p + sum_k N_k EE[n_k] - S - 1) log(1 - p) + "const".
$

#result(title: "Update: Beta")[
  $
    q^*(p) = "Beta"(tilde(a)_p, tilde(b)_p),
    quad tilde(a)_p = a_p + S,
    quad tilde(b)_p = b_p + sum_k N_k thin EE[n_k] - S
  $
  $tilde(a)_p$ counts the expected #super[13]C atoms, $tilde(b)_p$ the expected #super[12]C atoms.
]

// ============================================================
= Algorithm
// ============================================================

#figure(
  btable(
    columns: (auto, auto, 1fr),
    header: ([Factor], [Family], [Variational parameters]),
    $q(z_n, j_n)$, [categorical over $(k, j)$ and background], $r_(n k j), thin r_(n 0)$,
    $q(pi)$, [Dirichlet], $tilde(alpha)_0, dots, tilde(alpha)_K$,
    $q(mu_k, sigma_k^2)$, [Normal-inverse-gamma], $tilde(mu)_k, tilde(nu)_k, tilde(a)_k, tilde(b)_k$,
    $q(m_k | c_k)$, [Gaussian per charge], $tilde(m)_(k c), tilde(tau)_k^2$,
    $q(c_k)$, [categorical on ${1, 2, 3, 4}$], $q(c_k = c)$,
    $q(n_k)$, [categorical on a grid], $q(n_k = C)$,
    $q(p)$, [Beta], $tilde(a)_p, tilde(b)_p$,
    $q(s^2)$, [inverse-gamma], $tilde(a)_s, tilde(b)_s$,
  ),
  caption: [Variational factors and their parameters.],
)

#import "@preview/algorithmic:1.0.7"
#import algorithmic: style-algorithm, algorithm-figure
#show: style-algorithm
#algorithm-figure(
  "CAVI from random features",
  vstroke: .5pt + luma(200),
  {
    import algorithmic: *
    Procedure(
      "CAVI",
      ("pixels", "K", "J", "I"),
      {
        Assign[$w_n$][$I_n slash min_m I_m$]
        For(
          $k = 1, dots, K$,
          {
            Assign[$(m_(0 k), hat(m)_k)$][$(t_n, y_n)$ of a pixel $n tilde w$, with $y_n in [300, 2000]$]
            Assign[$c_k$][$tilde (0.1, 0.5, 0.3, 0.1)$]
          },
        )
        Assign[$q$][$p(theta)$]
        LineBreak
        For(
          $i = 1, dots, I$,
          {
            Assign[$r_(n k j), r_(n 0)$][softmax over $(k, j)$ and background of $log rho$]
            Assign[$N_k, N_0$][$sum_n w_n sum_j r_(n k j), thin sum_n w_n r_(n 0)$]
            Assign[$q(pi)$][$"Dir"(alpha + (N_0, dots, N_K))$]
            Assign[$q(mu_k, sigma_k^2)$][NIG update]
            Assign[$q(m_k | c_k = c)$][Gaussian update #h(0.4em) $q(c_k)$ #sym.arrow.l softmax of $ell_c$]
            Assign[$q(s^2)$][inverse-gamma update]
            Assign[$q(n_k)$][grid update #h(0.4em) $q(p)$ #sym.arrow.l Beta update]
            Assign[$cal(L)_i$][ELBO of $(r, q)$]
          },
        )
        Return[$q$, $(cal(L)_i)_i$]
      },
    )
  }
)

#figure(
  btable(
    columns: (auto, 1fr),
    header: ([Prior], [Value]),
    [position $(m_(0 k), hat(m)_k)$], [pixel drawn with probability $prop w_n$ (uniform over the window: `--init uniform`)],
    [charge], [$c_k tilde (0.1, 0.5, 0.3, 0.1)$ (centres the averagine $p(n_k)$); the prior over $c$ is the same for all $k$],
    [$tau_k$, $nu_(0 k)$, $EE[sigma_k^2]$], [$0.1$ Da, $0.1$, $(0.5 "min")^2$],
    [$EE[s^2]$, $alpha_k$, $K$], [$(0.05 "Da")^2$ (learned), $1$, $30$],
  ),
  caption: [Random start (`random_hyperparameters`).],
)

#remark(title: "Efficiency")[
  Only pixels inside a feature's bounding box ($kappa approx 5$ widths around the apex and each comb tooth) get
  non-zero responsibilities for that feature. Pixels outside all boxes are pure background and only contribute
  their total weight to $tilde(alpha)_0$. Recompute the boxes every few iterations, not within a sweep.
]

#remark(title: "Remark")[
   The core issue is that almost all ions will belong to the background so $z_n = 0$. We do not want to calculate CAVI updates for each of those update steps. So we do not want to consider each component $k$ for each isotope but only a fixed number of components $k_"eff"$ so 

  $
    sum_n sum_k sum_j ... = sum_n sum_(k_"eff") sum_n ...
  $

  We can achieve that, via selecting a number of potential candidates first fitting those and then fitting the rest of not easy to fit features.
]

// ============================================================
= ELBO
// ============================================================

$
  cal(L) = & sum_n w_n [sum_(k, j) r_(n k j)(log rho_(n k j) - log r_(n k j)) + r_(n 0)(log rho_(n 0) - log r_(n 0))] \
  & - D_"KL"(q(pi) || p(pi)) - D_"KL"(q(p) || p(p)) - D_"KL"(q(s^2) || p(s^2)) \
  & - sum_k [D_"KL"(q(mu_k, sigma_k^2) || p) + D_"KL"(q(m_k, c_k) || p) + D_"KL"(q(n_k) || p)]
$

#figure(
  btable(
    columns: (auto, 1fr),
    header: ([Factor], [KL divergence to the prior (`elbo.py`)]),
    $pi$, [Dirichlet: $log Gamma(sum tilde(alpha)) - sum log Gamma(tilde(alpha)_k) - dots + sum (tilde(alpha)_k - alpha_k)(psi(tilde(alpha)_k) - psi(sum tilde(alpha)))$],
    $(mu_k, sigma_k^2)$, [$D_"KL"("IG" || "IG") + 1/2 [log (tilde(nu)_k slash nu_(0 k)) + nu_(0 k) slash tilde(nu)_k - 1 + nu_(0 k) (tilde(mu)_k - m_(0 k))^2 tilde(a)_k slash tilde(b)_k]$],
    $(m_k, c_k)$, [$D_"KL"("Cat"(q(c_k)) || "Cat"(rho_k)) + sum_c q(c_k = c) D_"KL"(cal(N)(tilde(m)_(k c), tilde(tau)_k^2) || cal(N)(hat(m)_k, tau_k^2))$],
    $n_k$, [categorical on the grid],
    $p$, [Beta],
    $s^2$, [inverse-gamma (the gamma KL with shape $a$, rate $b$)],
  ),
  caption: [KL terms. $log rho$ is taken at the current $q$; each update raises $cal(L)$.],
)

// ============================================================
= Plots and results
// ============================================================

`make algorithm` (`plots/plot_cavi.py`) runs the algorithm on a spectrum and writes the plots below to `plots/cavi_real`.
Colours identify components in all panels (@fig-metrics, last page); only the $12$ components with most ions at the end are drawn. A component
counts as supported if $N_k = tilde(alpha)_k - alpha_k > 30$.

#figure(
  btable(
    columns: (auto, auto, 1fr),
    header: ([Plot], [Panel], [Meaning]),
    [`elbo.png`], [$cal(L)$ per sweep], [lower bound on the log evidence; flattening means converged],
    [], [$|Delta cal(L)|$, log], [size of each step; crosses mark decreases (`float32` noise)],
    [`metrics.png`], [1 expected ions $N_k$], [weighted number of pixels assigned to a component; dashed: support threshold],
    [], [2 supported components], [left: how many of the $K$ the data use; right: share of all ions explained by features, not background],
    [], [3 RT apex $mu_k$], [posterior mean elution apex and $plus.minus 2$ sd; dotted: reference RT of a matched component],
    [], [4 mass minus final], [convergence of $tilde(m)_(k c)$ (zero at the end by construction, not an accuracy); band: $plus.minus 2 tilde(tau)_k$],
    [], [5 peak width], [$sqrt(b_s slash (a_s - 1))$, one value for all; how close in m/z a pixel must be to a component],
    [], [6 #super[13]C probability $p$], [with $90 %$ interval; near $0.011$ the isotope teeth are explained by the comb, near $0$ by components of their own],
    [], [7 final $q(c_k)$], [charge posterior per component; a row equal to the prior is uninformed; crosses: reference charge],
    [], [8 final $q(n_k)$], [carbon-count posterior ($n approx 0.044 dot m dot c$); width: uncertainty; informative only where the comb is used],
    [`intensity.png`], [left], [fitted against reference intensity of each feature, log axes; dashed $y = x$, solid median ratio, dotted ceiling; colour: components per feature; title: Pearson (log), Spearman],
    [], [right], [the same as a ratio per feature, components above the bars, "none": no component],
    [`fit_full.mp4`], [top], [actual map, fitted expected map (posterior mean), posterior std of the expected counts; one frame per sweep, frame $0$ is the random start],
    [], [bottom], [projections on m/z and RT: actual (grey) against fitted (colour: std)],
    [`fit_component<k>.mp4`], [], [the same, zoomed on the component with most ions],
    [`..._3d.mp4`], [], [surface: fitted mean, colour: std, wireframe: actual counts],
  ),
  caption: [The plots of `plot_cavi.py`.],
)

*Fitted map.* With $W = sum_n w_n$, $P_(k j) = sum_n q(n_k = n) "Binom"(j | n, p)$, $mu_(k c j) = m_(k c) + j Delta slash c$,
the expected ions in the bin $[t_a, t_b] times [y_a, y_b]$ are
$
  W (pi_0 (Delta t Delta y) / (T Y) + sum_k pi_k thin A_k sum_j P_(k j) sum_c q(c_k = c) thin B_(k c j))
$
$
  A_k = Phi((t_b - mu_k) / sigma_k) - Phi((t_a - mu_k) / sigma_k), quad
  B_(k c j) = Phi((y_b - mu_(k c j)) / s) - Phi((y_a - mu_(k c j)) / s).
$
The std is taken over $12$ parameter draws from $q$ ($q(c)$, $q(n)$ marginalised): the uncertainty of the model, not the noise.

*Intensity.* The intensity of a reference feature is $min I$ times the expected ions of all components whose mass lies
on one of its isotope positions $m + j Delta slash c$ ($j < J$, reference charge) with RT within $1$ min (`cavi/evaluate.py`).
The ceiling is (file intensity) / (sum of reference intensities): the reference is a model estimate and can exceed the
pixel sum.


#figure(
  image("plots/cavi_real/intensity.png", width: 100%),
  caption: [`intensity.png`: fitted against reference intensity.],
) <fig-intensity>

#figure(
  btable(
    columns: (auto, auto, auto),
    header: ([Setting (OpenMS example map)], [reference features matched], [supported components]),
    [$K = 20$ / $30$ / $40$, six seeds], [$4.5$ / $5.5$ / $6.0$ of $8$], [$18$--$20$ / $26$--$29$ / $35$--$38$],
    [$K = 100$ / $200$], [$8$ / $8$], [-- / $131$],
    [intensity, $K = 30$], [Pearson (log) $0.84$--$0.97$], [fitted $0.32$--$0.49$ times the reference (ceiling $0.82$)],
    [uniform instead of pixel start], [$0$--$1$ of $8$], [broad width stays at $0.3$ Da],
  ),
  caption: [Results. Matched: a supported component at the monoisotopic mass and RT of a reference feature.],
)

#remark(title: "Limitations")[
  - *The comb is not used:* $p approx 2 dot 10^(-6)$ and $max_c q(c_k) = 0.50$ (the prior) for every seed and every $K$; each isotope tooth becomes a component.
  - *Few components cannot prune, many fragment:* $K = 200$ has $83$ of its $131$ supported components on isotope teeth; $alpha_k < 1$ does not help (it only acts on components with about one ion).
  - *Weights are not counts on real data:* the posterior is probably overconfident.
]

#page(flipped: true, margin: (x: 1.5cm, y: 1.8cm))[
  #figure(
    image("plots/cavi_real/metrics.png", width: 100%),
    caption: [`metrics.png`: posterior summaries over the sweeps (sweep $0$ is the random start).],
  ) <fig-metrics>
]
