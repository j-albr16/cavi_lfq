// ---------- theme ----------

#let accent = rgb("#1f4e79")
#let accent-light = rgb("#eef3f9")
#let note-color = rgb("#b8860b")
#let note-bg = rgb("#fbf7ea")

#let template(doc) = {
  set document(title: "Variational Bayes for LC–MS feature detection")
  set page(
    paper: "a4",
    margin: (x: 2.2cm, top: 2.5cm, bottom: 2.3cm),
    header: context {
      if counter(page).get().first() > 1 {
        set text(8.5pt, fill: luma(110))
        [_Variational Bayes for LC–MS feature detection_ #h(1fr) #counter(page).display()]
        v(-0.6em)
        line(length: 100%, stroke: 0.4pt + luma(180))
      }
    },
  )

  set text(font: "New Computer Modern", size: 10.5pt, lang: "en")
  set par(justify: true, leading: 0.62em)
  set heading(numbering: (..n) => if n.pos().len() <= 2 { numbering("1.1", ..n) })
  show heading.where(level: 1): it => {
    v(1.1em)
    set text(fill: accent, size: 14pt)
    it
    v(0.35em)
  }
  show heading.where(level: 2): it => {
    v(0.8em)
    set text(fill: accent, size: 11.5pt)
    it
    v(0.2em)
  }
  show heading.where(level: 3): it => {
    v(0.5em)
    set text(size: 10.5pt)
    it
  }
  show math.equation.where(block: true): set block(spacing: 0.9em)
  set list(indent: 0.6em)
  set enum(indent: 0.6em)
  doc
}

// ---------- helpers ----------
#let result(title: "Update", body) = block(
  width: 100%,
  fill: accent-light,
  stroke: (left: 2.5pt + accent),
  inset: (x: 12pt, y: 10pt),
  radius: (right: 3pt),
  breakable: false,
)[
  #text(fill: accent, weight: "bold", size: 9pt, tracking: 0.04em, upper(title))
  #v(-0.2em)
  #body
]

#let remark(title: "Note", body) = block(
  width: 100%,
  fill: note-bg,
  stroke: (left: 2.5pt + note-color),
  inset: (x: 12pt, y: 9pt),
  radius: (right: 3pt),
  breakable: false,
)[#text(weight: "bold", fill: note-color)[#title.] #body]

#let btable(columns: auto, align: left + horizon, header: (), ..rows) = table(
  columns: columns,
  stroke: none,
  inset: (x: 6pt, y: 4.5pt),
  align: align,
  table.hline(stroke: 0.9pt),
  table.header(..header.map(h => text(weight: "bold", h))),
  table.hline(stroke: 0.5pt),
  ..rows,
  table.hline(stroke: 0.9pt),
)

// graphical model: nodes (id, p: (x, y) in cm, label, obs), edges (from, to), plates
#let gm(nodes, edges, plates: (), width: 9.4cm, height: 4.7cm, r: 0.38) = {
  let u = 1cm
  let pos = (:)
  for nd in nodes { pos.insert(nd.id, nd.p) }
  box(width: width, height: height, {
    for pl in plates {
      place(dx: pl.x0 * u, dy: pl.y0 * u, rect(
        width: (pl.x1 - pl.x0) * u, height: (pl.y1 - pl.y0) * u,
        radius: 4pt, stroke: 0.6pt + luma(100),
      ))
      place(dx: (pl.x1 - 0.42) * u, dy: (pl.y1 - 0.45) * u, text(9pt, pl.label))
    }
    for e in edges {
      let (a, b) = e
      let (x1, y1) = pos.at(a)
      let (x2, y2) = pos.at(b)
      let dx = x2 - x1
      let dy = y2 - y1
      let l = calc.sqrt(dx * dx + dy * dy)
      let ux = dx / l
      let uy = dy / l
      let sx = x1 + r * ux
      let sy = y1 + r * uy
      let ex = x2 - r * ux
      let ey = y2 - r * uy
      let h = 0.19
      let w = 0.07
      let bx = ex - h * ux
      let by = ey - h * uy
      place(line(start: (sx * u, sy * u), end: (bx * u, by * u), stroke: 0.7pt))
      place(polygon(
        fill: black, stroke: none,
        (ex * u, ey * u),
        ((bx - w * uy) * u, (by + w * ux) * u),
        ((bx + w * uy) * u, (by - w * ux) * u),
      ))
    }
    for nd in nodes {
      let (x, y) = nd.p
      let obs = nd.at("obs", default: false)
      place(dx: (x - r) * u, dy: (y - r) * u, circle(
        radius: r * u, stroke: 0.8pt, fill: if obs { luma(205) } else { white },
      ))
      place(dx: (x - r) * u, dy: (y - r) * u, box(
        width: 2 * r * u, height: 2 * r * u, align(center + horizon, text(9.5pt, nd.label)),
      ))
    }
  })
}

