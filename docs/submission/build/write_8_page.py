import os
content = r"""\documentclass{zapiski}
\originfo{1}{}{}{2026}
\setcounter{page}{1}
\date{}

\usepackage[cp1251]{inputenc}
\usepackage[T2A]{fontenc}
\usepackage[russian, english]{babel}
\usepackage{times}
\usepackage{url}
\usepackage{latexsym}
\usepackage{graphicx}
\usepackage{amssymb}
\usepackage{amsmath}
\usepackage{hyperref}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{caption}
\usepackage{cite}
\usepackage{xcolor}
\usepackage{array}
\usepackage{float}
\usepackage{multirow}
\restylefloat{table}

\renewcommand{\UrlFont}{\ttfamily\small}

\english

\begin{document}

\title[Latent Trajectories in Recurrent Depth]{What the Latent
Trajectory of a Recurrent-Depth Transformer Encodes: Geometry, State-Tracking, and Formatting}

\author[Varaksin, Shiianov, Sverdlov]{
  \textbf{Arsenii Varaksin},
  \textbf{Alexander Shiianov},
  \textbf{David Sverdlov}
\\
  \small{
    \textbf{Mentor:} Serguei Barannikov (Skoltech)
  }
\\
  \small{
    \textbf{Correspondence:} \href{mailto:varbarn22@gmail.com}{varbarn22@gmail.com}
  }
}

\begin{abstract}
A recurrent-depth transformer's latent trajectory is usually read as a trace of its reasoning.
Measuring that reading directly on Huginn-3.5B, we find that the trajectory's most conspicuous geometric
structure is selected by the surface form of the instruction rather than by the task, and that
much of what we had scored as capability was answer formatting. The recurrent map has a
sharp qualitative structure: a single instruction word switches it between a settling regime and a
damped period-6 rotation, causally controlled by one row of the embedding matrix. However, 
first-token scoring reads 0.125 where the answer is actually present in 0.781 of
generations; supplying the output format reproduces published benchmarks. The regime
and the answer occupy non-overlapping windows of the same trajectory, with the answer fixed 
before the regime becomes measurable. Furthermore, evaluating the trajectory's winding and 
steps-to-settle across algorithmic tasks reveals that multi-step logic does not induce looping 
(refuting naive depth-encoding hypotheses), whereas state retention drives the number of steps to settle. 
Finally, we reproduce two-scale dynamics on real data, demonstrating that apparent orthogonality 
is a methodological artifact of unnormalized steps. We report the claims this project
withdrew alongside those it kept, ensuring methodological rigor.
\end{abstract}

\keywords{recurrent-depth transformers, latent reasoning, interpretability,
evaluation protocols, dynamical systems, state-tracking}

\maketitle

\section{Introduction}

A recurrent-depth transformer reuses one weight-tied block many times at inference, so the
sequence of latent states it passes through is often read as a trace of reasoning --- something
whose shape should reflect the computation being performed. We tested that reading on
Huginn-3.5B \cite{geiping2025huginn} and report three main dimensions of findings.

The first is that the map has real, sharp, causally controllable structure, but that this
structure is selected by the surface form of the instruction rather than by the task. The second
is that a substantial part of our own measured capability record --- and, we argue, of the
standard way such models are scored --- was measuring whether the model emits its answer in the
position a scorer reads, not whether it has the answer. The third evaluates specific hypotheses about state-tracking and depth-encoding, demonstrating that steps-to-settle act as a proxy for effective compute, while winding numbers do not encode reasoning depth.

Our work builds upon several key contributions. Geiping et al. \cite{geiping2025huginn} introduced Huginn and observed latent orbits primarily on question and digit tokens. Blayney et al. \cite{blayney2026mechanistic} found non-fixed-point behavior in 0.02--2.8\% of tokens in Huginn-0125. Pappone et al. \cite{pappone2025twoscale} proposed two-scale latent dynamics with spiral trajectories and orthogonality metrics. Merrill et al. \cite{merrill2024illusion} demonstrated that the "state" in state-space models is an illusion, while Grazzi et al. \cite{grazzi2024statetracking} proved that finite-precision LRNNs with positive eigenvalues cannot solve parity tasks. 

Concretely, this report contributes:
\begin{itemize}\itemsep2pt \parskip0pt
\item a validated scalar instrument for latent rotation, with a threshold that reproduces hand-assigned regime labels on 691 of 696 orbits;
\item the finding that this regime is selected by one instruction word and causally controlled by the embedding matrix;
\item a decomposition of measured capability into answer availability and answer formatting;
\item a timing result placing the regime and the answer in non-overlapping windows (Figure~\ref{fig:onset});
\item experimental refutation of naive depth-encoding hypotheses and support for state-retention hypotheses via length-matched controls;
\item a critical refutation of two-scale orthogonality dynamics, showing they are methodology-dependent artifacts.
\end{itemize}

\begin{figure}[t]
\centering
\includegraphics[width=0.86\linewidth]{pic/onset.pdf}
\caption{Rotation power in a sliding 12-unroll window, median over six rotating and six settling prompts.}
\label{fig:onset}
\end{figure}

\section{Model, Protocol, and Metrics}

Huginn-3.5B (\texttt{tomg-group-umd/huginn-0125}, revision \texttt{bb6621b6\ldots}) has the structure:
$\text{embed} \rightarrow \text{prelude}(2) \rightarrow \bigl[\,\text{core\_block}(4)\,\bigr]^{\times r} \rightarrow \text{coda}(2) \rightarrow \text{ln\_f} \rightarrow \text{lm\_head}$, where the prelude output $e$ is re-injected at every unroll. All measurements are inference-only in float32 and hook the output of \texttt{core\_block[-1]}. Unless stated, depth is $r = 32$ or $r = 48$.

\begin{table}[h]
\centering
\small
\caption{Experimental datasets and configurations}
\begin{tabular}{llll}
\toprule
Task & Levels & N (trajectories) & Prompt format \\
\midrule
PARARULE-Plus & depths 2--5 & 10 / depth & "Question: ... Answer:" \\
Counting ($\pm1$ sum) & lengths 2--32 & 5--15 / length & "Start at 0. Add 1. Subtract 1. ..." \\
Parity (switch) & lengths 4--48 & 10 / length & "Light is off. Flip. Wait. ..." \\
Running-max & lengths 4--48 & 8 / length & "Numbers: 3 7 2 ... Largest so far?" \\
FineWeb (real texts) & -- & 1000 & -- \\
\bottomrule
\end{tabular}
\end{table}

We compute multiple metrics per trajectory:
\begin{itemize}
    \item \textbf{Rotation Power ($R$):} Fraction of non-DC power in the period-6 bin over a 24-unroll tail (5280-D).
    \item \textbf{Winding number:} Accumulated angle in 2D-PCA projection, normalized to $2\pi$.
    \item \textbf{Steps-to-settle:} First step where $\|\Delta h_t\| < 0.1$, serving as a proxy for effective computation.
    \item \textbf{Two-Scale Metrics:} Acceleration ($a(k) = \|\Delta(k) - \Delta(k-1)\|_2$) and Orthogonality ($\cos\theta$).
    \item \textbf{Persistent homology H1:} Maximum normalized persistence computed via \texttt{ripser}.
\end{itemize}

\section{The Recurrent Map Has a Sharp, Causally Controlled Regime}

\subsection{Measuring rotation without projecting}
Taking the discrete Fourier transform along the unroll axis, in the full 5280-dimensional state space and without any projection, we define the \emph{rotation power} $R$ as the fraction of non-DC power falling in the period-6 bin over a 24-unroll tail. This instrument is self-consistent to $4\times10^{-8}$. A single threshold, $R > 0.6677$, reproduces the hand-assigned binary label on 691 of 696 orbits and on 688 held out leave-one-bank-out. 

\subsection{One instruction word switches the regime}
Holding the sequence, the required answer and the token count fixed at 53, the instruction \emph{``report the largest \textbf{symbol}''} produces a damped period-6 rotation in 36 of 36 paired cells, while \emph{``largest \textbf{element}''} produces settling in 0 of 36. This is causally controlled by a single row of the embedding matrix; editing the row flips the regime exactly.

\subsection{No hysteresis, and the contraction rate predicts a timescale}
Swapping $e$ mid-trajectory, 62 of 64 switches take: the regime follows the parameter the map currently has rather than the trajectory's history. Entering rotation takes a median of 15.5 unrolls and leaving it 1.0. The entering time brackets a prediction registered before the run from the measured contraction rate $\rho \approx 0.83$, namely $\ln(0.05)/\ln(0.83) = 16.1$ unrolls.

\section{What the Capability Measurements Were Measuring}

First-token exact match reads 0.125 where the answer is actually produced in 0.781 of generations, against a \emph{measured} cross-item false-positive rate of 0.042--0.152. The token that displaces the gold is \emph{The}: it opens 39.4\% of 1260 banked generations. Depth makes this worse: across $r = 2,4,8,16,32$ the rate of opening with \emph{The} runs $0.032 \rightarrow 0.647$, while the rate of opening with the gold does not move ($p = 0.515$).

The published ARC-Easy figure for this model is 0.699 at $r = 32$. Our bare-prompt, zero-shot arm (matching their protocol) reads \textbf{0.658}. Independently, five in-context examples under option-letter argmax give \textbf{0.723} ($p = 3.35\mathrm{e}{-}07$). Measuring the trajectory at every generated position, the forward pass that carries the answer is deeper than position 0 ($4.07 \rightarrow 7.89$, $p = 0.0031$) and deeper than the surrounding prose positions.

\section{The Regime and the Answer Occupy Different Windows}

The rotation statistic is computed over the tail of the trajectory; the answer, by contrast, is best-ranked at a median unroll of about four. Over the decision window (unrolls 0--11), rotating and settling prompts are indistinguishable: $0.2292$ against $0.2296$. Over the tail the same prompts separate completely, $0.862$ against $0.298$. 

Because the answer is fixed ~12 unrolls before the regime becomes measurable, behavioural nulls about the regime manipulate a property that had not yet come into existence when the readout was settled. \textbf{This is a timing claim rather than a causal one.} It does not show that the regime could not affect behaviour, only that the answer is fixed before the regime becomes measurable.

\section{State Tracking and Winding: Evaluating the Hypotheses}

We evaluated three hypotheses regarding the relationship between trajectory shape and reasoning:
\begin{itemize}
    \item[H1] \textbf{Settle / loop / drift.} All answer-token trajectories exhibit exponential contraction. Unbounded drift is excluded by construction (normalisation sphere of radius $76.386$).
    \item[H2] \textbf{Loops encode reasoning depth.} When the path loops, its winding number grows with the number of reasoning steps required.
    \item[H3] \textbf{Contraction prevents state tracking.} Strict contraction prevents the model from maintaining a running count, requiring looping or drifting for stateful tasks.
\end{itemize}

\subsection{E1: PARARULE-Plus — Winding Does Not Track Depth (H2 Refuted)}
All 40 answer-token trajectories (depths 2--5) exhibit settling behavior. Correlation analysis reveals that winding vs depth (per-row, N=40) yields $\rho = -0.037$, $p = 0.82$. Multi-step symbolic logic does not induce looping; winding-depth correlation is null. Furthermore, H2 is architecturally untestable on this model regarding \emph{adaptive} depth: Huginn has no halting head, so every position receives exactly $r$ unrolls with nothing to allocate.

\subsection{E4: Phase Map and Force-Loop}
With reduced budget (\textit{num\_steps}=16), loops appear ($n_{\text{ops}}=24$: 7/8 loop). With $\textit{num\_steps} \ge 24$, all trajectories settle. Loops are a symptom of insufficient computation budget, not depth per se.

\subsection{E2 and E3: Synthetic Counting and Length-Matched Controls (H3 Supported)}
Running $\pm1$ sum (length $n_{\text{ops}} \in \{2\ldots 32\}$): Steps-to-settle vs $n_{\text{ops}}$ yields $\rho = 0.718$, $p < 1e-5$. The winding correlation is driven by prompt length, not state tracking. To isolate this, E3 used \textit{Track} and \textit{local} tasks sharing identical length. At identical length, \textit{Track} yields steps vs $n_{\text{ops}}$ correlation of $+0.558$ ($p < 1e-8$), while \textit{Local} yields $-0.576$. The sign of the correlation is determined solely by task type: state retention increases computation. This constitutes strong support for H3. This generalizes to Parity (E5), where steps-to-settle correlates with length ($\rho = 0.615$).

\section{Two-Scale Dynamics and Orthogonality}

\subsection{Two-Scale Dynamics on Synthetic Data}
Applying Pappone et al.'s two-scale metrics to synthetic data (depths 2--5), we find that Acceleration grows with depth ($4.89 \rightarrow 7.45$, $\rho = 0.997$), providing a strong signal where Winding fails. Orthogonality remains stable at $\sim -0.5$.

\subsection{Refutation of Pappone et al. on Real Data}
During reproduction of two-scale dynamics on real FineWeb text data, we obtained results that \textbf{fundamentally differ} from Pappone et al. (2025). 
During the reproduction experiment, we discovered:
\begin{enumerate}
    \item \textbf{Without step normalization, orthogonality is not stably achieved.} The orthogonality reported by Pappone et al. is an artifact of failing to normalize the step vectors, confounding orthogonality with step norm (correlation $-0.0436$ after normalization).
    \item \textbf{Alternative dynamics type discovered.} Instead of spiral dynamics ($\cos \approx 0.5$--0.65), we observe oscillatory dynamics ($\cos \approx -0.687$).
\end{enumerate}
This demonstrates that the dynamics type depends heavily on measurement methodology.

\section{Reliability and Known Limitations}

We report the claims this project withdrew alongside those it kept. 
\begin{itemize}
    \item A linear probe of the class used for several of our nulls \textbf{cannot recover a label that is a guaranteed deterministic function of its own input features} (0.690 against a 0.600 baseline). Those nulls are uninformative rather than negative.
    \item The \textbf{extraction rule dominates the number}: exact match recovers 0.038 of the answers and a last-number rule 0.240 at half the false-positive rate.
    \item \textbf{Small sample sizes.} Insufficient power to detect rare phenomena (0.02--2.8\% per Blayney).
    \item \textbf{Proxy metrics.} 2D-PCA winding is sign-arbitrary; steps-to-settle conflates small steps with convergence.
\end{itemize}

\section{Conclusion}

The recurrent map of Huginn-3.5B carries a sharp qualitative structure selected by surface form. Our behavioural tests of that structure came back null, but the timing result shows they intervened on a property that had not yet become measurable when the answer was already fixed. A substantial part of capability evaluations was merely the model's formatting willingness. 
Furthermore, answer-token trajectories converge to fixed points. The model fails synthetic counting, consistent with theoretical predictions. No winding-depth signal is detected, refuting naive H2, but \textbf{steps-to-settle is the effective compute proxy} (supporting H3). Finally, two-scale orthogonality results are methodology-dependent artifacts of unnormalized steps.

\section{Contribution}

\begin{table}[h]
\centering
\caption{Contribution matrix}
\begin{tabular}{p{4cm}p{3cm}p{3.5cm}p{3cm}}
\toprule
Component & Arsenii Varaksin & Alexander Shiianov & David Sverdlov \\
\midrule
Formatting confound \& Timing hinge & \checkmark & -- & -- \\
Rotation\_power statistic & \checkmark & -- & -- \\
Steering and geometric regime control & \checkmark & -- & -- \\
Extraction pipeline & -- & \checkmark & -- \\
Winding \& Steps-to-settle & -- & \checkmark & -- \\
Shape class \& Persistent homology & -- & \checkmark & -- \\
PARARULE \& Counting exps & -- & \checkmark & -- \\
Two-scale dynamics & -- & -- & \checkmark \\
FineWeb orthogonality reproduction & -- & -- & \checkmark \\
\bottomrule
\end{tabular}
\end{table}

\textbf{Arsenii Varaksin (34\%):} Discovered the formatting evaluation confound. Defined and validated the scalar \texttt{rotation\_power} statistic. Established the timing hinge separating regime from answer computation. Executed causal embedding steering experiments. Drafted sections 3--5.

\textbf{Alexander Shiianov (33\%):} Developed trajectory extraction infrastructure. Implemented MVP metrics (winding, steps-to-settle, homology). Conducted experiments E1--E5 across PARARULE-Plus, counting, and length-matched controls, refuting naive H2 and supporting H3.

\textbf{David Sverdlov (33\%):} Implemented two-scale dynamics metrics. Conducted reproduction of orthogonality on FineWeb data. Demonstrated that without step normalization, apparent orthogonality is a methodological artifact. Validated acceleration as a depth indicator.

\textbf{Serguei Barannikov (Mentor):} Problem formulation, hypotheses H1--H3, and supervision.

\begin{thebibliography}{9}

\bibitem{geiping2025huginn}
Jonas Geiping et~al., \emph{Scaling up test-time compute with latent reasoning: a recurrent depth
approach}, arXiv:2502.05171, 2025.

\bibitem{blayney2026mechanistic}
E. Blayney, et al. "A Mechanistic Analysis of Looped Reasoning Language Models." arXiv:2604.11791, 2026.

\bibitem{pappone2025twoscale}
Francesco Pappone, Donato Crisostomi, and Emanuele Rodol\`a, \emph{Two-scale latent dynamics for
recurrent-depth transformers}, arXiv:2509.23314, 2025.

\bibitem{merrill2024illusion}
W. Merrill, J. Petty, and A. Sabharwal, "The Illusion of State in State-Space Models." arXiv:2404.08819, 2024.

\bibitem{grazzi2024statetracking}
Riccardo Grazzi et~al., \emph{Unlocking state-tracking in linear RNNs through negative eigenvalues}, arXiv:2411.12537, 2024.

\bibitem{graves2016act}
Alex Graves, \emph{Adaptive computation time for recurrent neural networks}, arXiv:1603.08983, 2016.

\bibitem{banino2021pondernet}
Andrea Banino, Jan Balaguer, and Charles Blundell, \emph{PonderNet: learning to ponder}, arXiv:2107.05407, 2021.

\bibitem{dehghani2019ut}
Mostafa Dehghani et~al., \emph{Universal transformers}, arXiv:1807.03819, 2018.

\bibitem{bai2019deq}
Shaojie Bai, J.~Zico Kolter, and Vladlen Koltun, \emph{Deep equilibrium models}, arXiv:1909.01377, 2019.

\end{thebibliography}

\end{document}
"""
with open('merged_paper.tex', 'w', encoding='cp1251', errors='replace') as f:
    f.write(content)
