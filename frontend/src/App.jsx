import {
  Activity,
  ArrowDownRight,
  Atom,
  BrainCircuit,
  ChartNoAxesCombined,
  ChevronRight,
  FlaskConical,
  Code2,
  Layers3,
  Microscope,
  Waves,
} from 'lucide-react'

import DitherVeil from './components/DitherVeil'
import './App.css'

const metrics = [
  {
    label: 'Spatial model',
    value: '2D',
    detail: 'Two-dimensional concentration field',
    icon: Layers3,
  },
  {
    label: 'Numerical method',
    value: 'FDM',
    detail: 'Finite Difference Method',
    icon: Activity,
  },
  {
    label: 'AI model',
    value: 'PINN',
    detail: 'Physics-Informed Neural Network',
    icon: BrainCircuit,
  },
  {
    label: 'PINN relative L2 error',
    value: '12.42%',
    detail: 'At simulation time 1800 seconds',
    icon: ChartNoAxesCombined,
  },
]

const stages = [
  {
    number: '01',
    title: 'Diffusion',
    description:
      'Model how a drug concentration spreads through a two-dimensional medium over time.',
    icon: Waves,
  },
  {
    number: '02',
    title: 'Reaction',
    description:
      'Include a reaction or decay term to represent the reduction in drug concentration.',
    icon: FlaskConical,
  },
  {
    number: '03',
    title: 'Prediction',
    description:
      'Train a physics-informed neural network to approximate the PDE solution.',
    icon: BrainCircuit,
  },
]

function App() {
  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="#overview" aria-label="PDE Lab home">
          <span className="brand-mark">
            <Atom size={21} strokeWidth={1.7} />
          </span>
          <span>
            <strong>PDE LAB</strong>
            <small>SCIENTIFIC COMPUTING</small>
          </span>
        </a>

        <nav className="main-nav" aria-label="Main navigation">
          <a className="active" href="#overview">Overview</a>
          <a href="#model">PDE Model</a>
          <a href="#pinn">PINN</a>
          <a href="#comparison">Comparison</a>
        </nav>

        <a
          className="github-link"
          href="https://github.com/anky2247/PDE-Drug-Diffusion"
          target="_blank"
          rel="noreferrer"
        >
          <Code2 size={17} />
          <span>Source code</span>
        </a>
      </header>

      <section className="hero" id="overview">
        <div className="hero-visual">
          <DitherVeil
            src="https://images.unsplash.com/photo-1737071371043-761e02b1ef95?q=80&w=1400&auto=format&fit=crop"
            fit="contain"
            pattern="floyd"
            pixelSize={2}
            levels={2}
            palette="duotone"
            inkColor="#120f17"
            paperColor="#f4f1ea"
            contrast={1.15}
            brightness={0}
            revealRadius={200}
            softness={0.6}
            linger={1}
            rimColor="#a78bfa"
            rim={0}
            reverse={false}
            wander={false}
            clickBurst
          />

          <div className="hero-shade" />

          <div className="hero-content">
            <div className="eyebrow">
              <span className="status-dot" />
              COMPUTATIONAL DRUG DELIVERY RESEARCH
            </div>

            <h1>
              PDE Drug
              <br />
              <span>Diffusion</span> Simulator
            </h1>

            <p className="hero-description">
              A computational framework for modelling drug diffusion and
              reaction using numerical methods and physics-informed deep
              learning.
            </p>

            <div className="hero-actions">
              <a className="primary-button" href="#model">
                Explore the model
                <ChevronRight size={17} />
              </a>
              <a className="secondary-button" href="#comparison">
                View results
                <ArrowDownRight size={17} />
              </a>
            </div>
          </div>

          <div className="hero-caption">
            <span>01 / COMPUTATIONAL MODEL</span>
            <span>DIFFUSION · REACTION · LEARNING</span>
          </div>
        </div>
      </section>

      <section className="metrics-section">
        <div className="section-heading">
          <div>
            <span className="section-kicker">PROJECT SNAPSHOT</span>
            <h2>Research at a glance</h2>
          </div>
          <span className="section-index">OVERVIEW / 001</span>
        </div>

        <div className="metrics-grid">
          {metrics.map((metric) => {
            const Icon = metric.icon

            return (
              <article className="metric-card" key={metric.label}>
                <div className="metric-top">
                  <span>{metric.label}</span>
                  <Icon size={19} />
                </div>
                <strong>{metric.value}</strong>
                <p>{metric.detail}</p>
              </article>
            )
          })}
        </div>
      </section>

      <section className="model-section" id="model">
        <div className="section-heading">
          <div>
            <span className="section-kicker">MATHEMATICAL FOUNDATION</span>
            <h2>The diffusion-reaction equation</h2>
          </div>
          <span className="section-index">MODEL / 002</span>
        </div>

        <div className="equation-panel">
          <div className="equation-copy">
            <span className="panel-label">GOVERNING PDE</span>
            <div className="equation">
              <span>∂C</span>
              <span className="fraction-line">=</span>
              <span>
                D(
                <span className="fraction">
                  <span>∂²C</span>
                  <span>∂x²</span>
                </span>
                +
                <span className="fraction">
                  <span>∂²C</span>
                  <span>∂y²</span>
                </span>
                ) − kC
              </span>
            </div>
            <p>
              The equation describes how concentration changes through
              spatial diffusion and first-order reaction.
            </p>
          </div>

          <div className="equation-legend">
            <div><span className="legend-symbol">C</span><span>Drug concentration</span></div>
            <div><span className="legend-symbol">D</span><span>Diffusion coefficient</span></div>
            <div><span className="legend-symbol">k</span><span>Reaction rate constant</span></div>
          </div>
        </div>
      </section>

      <section className="workflow-section" id="pinn">
        <div className="section-heading">
          <div>
            <span className="section-kicker">COMPUTATIONAL PIPELINE</span>
            <h2>From physics to prediction</h2>
          </div>
          <span className="section-index">METHOD / 003</span>
        </div>

        <div className="workflow-grid">
          {stages.map((stage) => {
            const Icon = stage.icon

            return (
              <article className="workflow-card" key={stage.number}>
                <div className="workflow-top">
                  <span>{stage.number}</span>
                  <Icon size={22} />
                </div>
                <h3>{stage.title}</h3>
                <p>{stage.description}</p>
              </article>
            )
          })}
        </div>
      </section>

      <section className="results-section" id="comparison">
        <div className="results-copy">
          <span className="section-kicker">MODEL PERFORMANCE</span>
          <h2>Numerical solution meets deep learning.</h2>
          <p>
            The project compares a finite-difference solution with a
            physics-informed neural network. The current PINN V2 experiment
            reports a relative L2 error of 12.42% at 1800 seconds.
          </p>
          <a
            className="text-link"
            href="https://pde-drug-diffusion.streamlit.app/"
            target="_blank"
            rel="noreferrer"
          >
            Open interactive Streamlit app
            <ChevronRight size={17} />
          </a>
        </div>

        <div className="result-stat">
          <span className="panel-label">PINN V2 · 1800 SECONDS</span>
          <strong>12.42<span>%</span></strong>
          <div className="result-bar">
            <span />
          </div>
          <small>Relative L2 error</small>
          <div className="result-foot">
            <span>Finite Difference Method</span>
            <span>PINN</span>
          </div>
        </div>
      </section>

      <footer className="footer">
        <a className="brand footer-brand" href="#overview">
          <span className="brand-mark">
            <Microscope size={19} />
          </span>
          <span>
            <strong>PDE LAB</strong>
            <small>SCIENTIFIC COMPUTING</small>
          </span>
        </a>
        <p>Built for computational modelling and scientific exploration.</p>
        <a href="https://github.com/anky2247/PDE-Drug-Diffusion" target="_blank" rel="noreferrer">
          GitHub <ArrowDownRight size={14} />
        </a>
      </footer>
    </main>
  )
}

export default App