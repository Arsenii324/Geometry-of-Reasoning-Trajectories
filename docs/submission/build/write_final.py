# -*- coding: utf-8 -*-
import os

latex_content = r"""\documentclass{zapiski}
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

\title[Latent Trajectories in Recurrent Depth]{What the Latent Trajectory of a Recurrent-Depth Transformer Encodes, and What Our Measurements Were Actually Measuring}

\author[Shiianov, Sverdlov, Varaksin]{
\textbf{Alexander Shiianov},
\textbf{David Sverdlov},
\textbf{Arsenii Varaksin}
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
A recurrent-depth transformer's latent trajectory is usually read as a trace of its reasoning. We find that most of what such readings measure is the model's willingness to answer in the format being scored. First, the recurrent map has sharp qualitative structure: one instruction word switches it between a settling regime and a rotation, causally controlled by the embedding matrix. Second, attempting to connect that structure to behaviour revealed that first-token scoring reads 0.125 where the answer is actually present in 0.781 of generations. Third, evaluating the trajectory's winding and steps-to-settle across algorithmic tasks reveals that state retention drives the number of steps to settle, while multi-step logic does not induce looping. Furthermore, we reproduce two-scale dynamics on real data, demonstrating that apparent orthogonality is a methodological artifact. We report the claims this project withdrew alongside those it kept, ensuring methodological rigor.
\end{abstract}

\keywords{recurrent-depth transformers, latent reasoning, interpretability, evaluation methodology, dynamical systems, test-time compute}

\maketitle

\section{Introduction}

A recurrent-depth transformer reuses one weight-tied block many times at inference, so the sequence of latent states it passes through is often read as a trace of reasoning. We tested that reading on Huginn-3.5B \cite{geiping2025scaling}. Adaptive-computation architectures measure per-position ponder time and find it rising with difficulty \cite{dehghani2019ut}. Each token's latent path falls into regimes---settling, looping, or drifting (H1). When the path loops, its winding number grows with the number of reasoning steps required by the task (H2). If the update is strictly contractive, the model cannot maintain a running count or memory, requiring looping or drifting for stateful tasks (H3).

Our null results on this axis were measuring the only quantity the model can vary---the global iteration count---on an architecture with no mechanism to vary it per instance. Huginn has no halting head: $\textit{num\_steps} = r$ unrolls the whole sequence together, so every token position receives exactly $r$ iterations and there is nothing to allocate. Furthermore, single-seed initializations, proxy metrics like 2D-PCA winding, and the uncorrected surface-form bias of first-token evaluation constrain the scope of any findings.

We set out to measure the geometry of latent reasoning, found that most of what we were measuring was output format, and fixing that changed what the geometry results mean. We evaluated four task families: PARARULE-Plus, synthetic counting, parity, and running-max. 

This report presents a verified scalar instrument for latent rotation, a causal timing result decoupling the regime from answer generation, length-matched controls supporting state-retention hypotheses, and a critical refutation of two-scale orthogonality. 

\section{The Rotation Regime and Output Formatting}

Measuring the same rotation statistic over two windows of the same orbit separates two facts that had been conflated. Over the decision window---unrolls 0--11, the earliest a period-6 orbit is definable at all, since the statistic requires two cycles---rotating and settling prompts are indistinguishable (0.229 against 0.230, separation 0.000). Over the tail they separate completely (0.862 against 0.298). A sliding window dates the divergence to between unrolls 12 and 16 (Figure \ref{fig:onset}). The early window is not quiet: its non-DC power exceeds the tail's by about two orders of magnitude. It is energetic and simply has no period-6 structure. Since the answer is best-ranked at a median unroll of about four, the earlier finding that a causally-induced regime change alters almost nothing behaviourally is not evidence that the regime is a dynamical epiphenomenon; it manipulated a property that had not yet come into existence when the readout was settled. \textbf{This is a timing claim rather than a causal one.} It does not show that the regime could not affect behaviour, only that on this architecture the answer is fixed roughly twelve unrolls before the regime becomes measurable.

\begin{figure}[t]
\centering
\includegraphics[width=0.7\textwidth]{pic/onset.pdf}
\caption{Rotation power in a sliding 12-unroll window, median over six rotating and six settling prompts. The x-axis $s$ represents the window's START, such that the window spans $[s, s+12)$.}
\label{fig:onset}
\end{figure}

The recurrent map's sharp qualitative structure is selected by the instruction word: ``report the largest \textbf{symbol}'' rotates 36/36, while ``largest \textbf{element}'' rotates 0/36. This selector is not semantic, not form, not tokenisation: ``symbol'''s semantic neighbours rotate 0/144, ``element'''s rotate 96/120, and form-matched controls rotate 72/72. Furthermore, rotating prompts rotate at 62.0\% of positions, whereas settling prompts are at 0.000 with no overlap. Swapping the injected parameter mid-trajectory re-aims the dynamics with no hysteresis. 

Scoring the first generated token, the model answers correctly on 12.5\% of items. Reading the generated text, the answer is present in 78.1\%---against a measured cross-item chance rate of 0.042 to 0.152. What occupies the first position is prose: the token \emph{The} opens 39.4\% of 1260 banked generations. Depth makes this worse rather than better---the rate of opening with \emph{The} rises twentyfold from $r = 2$ to $r = 32$ while the rate of opening with the answer does not move at all ($p = 0.515$). \textbf{The model is not failing to compute the answer; it is failing to put the answer where our metric looks.} Accuracy moves $0.416 \rightarrow 0.723$ with 35 items fixed against 4 broken when prompt format changes. The model's published ARC-Easy score is 69.9\% at $r = 32$. Read from the released code, that number is produced by a bare continuation prompt with no option list. Scored that way we obtain \textbf{65.8\%}. A second and independent route reaches the same place: five in-context examples under option-letter argmax give \textbf{72.3\%}. We applied a difference-of-means steering vector in the map's injected parameter. It reproduced neither the behavioural benefit of in-context examples nor any change in the rotation regime (0.000 to 0.080 accuracy; R moves by at most 0.0065). We read this as the wrong injection site rather than as two orthogonal directions.

\section{Hypothesis Testing on Algorithmic Tasks}

We use four task families in inference-only mode:

\begin{table}[H]
\centering
\caption{Experimental datasets and configurations}
\begin{tabular}{llll}
\toprule
Task & Levels & N (trajectories) & Prompt format \\
\midrule
PARARULE-Plus (qbao775) & depths 2--5 & 10 / depth & "Question: ... Answer:" \\
Counting ($\pm1$ sum) & lengths 2--32 & 5--15 / length & "Start at 0. Add 1. Subtract 1. ..." \\
Parity (switch) & lengths 4--48 & 10 / length & "Light is off. Flip. Wait. ..." \\
Running-max & lengths 4--48 & 8 / length & "Numbers: 3 7 2 ... Largest so far?" \\
FineWeb (real texts) & -- & 1000 & -- \\
\bottomrule
\end{tabular}
\end{table}

\subsection{E1: PARARULE-Plus --- Winding Does Not Track Depth}

All 40 answer-token trajectories (depths 2--5, n=10 per depth) exhibit settling behavior. Correlation analysis reveals:
\begin{itemize}
    \item Winding $\sim$ depth (per-level, N=4): $\rho = +0.20$, n.s. (threshold $\approx 1.0$).
    \item Winding $\sim$ depth (per-row, N=40): $\rho = -0.037$, $p = 0.82$.
    \item Partial winding $\sim$ depth $|$ length: $\rho = -0.234$, $p = 0.15$.
    \item Steps-to-settle: $20.2 \rightarrow 22.0$ across depths (weak growth).
\end{itemize}
\textbf{Conclusion:} Multi-step symbolic logic does not induce looping; winding-depth correlation is null.

\subsection{E2: Synthetic Counting --- H3 Support}

Running $\pm1$ sum, length $n_{\text{ops}} \in \{2\ldots 32\}$, 5 seeds. All trajectories settle, but:
\begin{itemize}
    \item Steps-to-settle $\sim n_{\text{ops}}$: $\rho = 0.718$, $p < 1e-5$ --- effective computation grows with state length.
    \item $|$winding$| \sim n_{\text{ops}}$: $\rho = 0.554$, $p = 0.002$, but partial by length yields $\rho = 0.04$, $p = 0.82$.
\end{itemize}
The winding correlation is driven by prompt length, not state tracking. This necessitates length-matched controls.

\subsection{E3: Length-Matched Control --- The Killer Experiment}

\textit{Track} and \textit{local} tasks share identical body text and differ only in the question ("What is the sum?" vs. "What was the last instruction?"), ensuring identical length. With 15 seeds:

\begin{table}[H]
\centering
\caption{Length-matched control results (15 seeds)}
\begin{tabular}{lccc}
\toprule
Condition & Metric & $\rho$ & p-value \\
\midrule
\multirow{2}{*}{Track (requires accumulation)} & steps $\sim n_{\text{ops}}$ & +0.558 & $< 1e-8$ \\
 & winding $\sim n_{\text{ops}}$ & +0.324 & 0.002 \\
\multirow{2}{*}{Local (last step only)} & steps $\sim n_{\text{ops}}$ & -0.576 & $< 1e-8$ \\
 & winding $\sim n_{\text{ops}}$ & +0.054 & 0.61 (n.s.) \\
\bottomrule
\end{tabular}
\end{table}

At identical length, the sign of the correlation is determined solely by task type: state retention increases computation; pure retrieval decreases it. This constitutes strong support for H3.

\subsection{E4: Phase Map and Force-Loop}

With reduced budget (\textit{num\_steps}=16), loops appear ($n_{\text{ops}}=24$: 7/8 loop; $n_{\text{ops}}=48$: 5/8 loop). With $\textit{num\_steps} \ge 24$, all trajectories settle. The phase map (\textit{num\_steps} $\times n_{\text{ops}}$) shows that the fraction of "non-settled" trajectories grows with constrained budget and longer counts. \textbf{Conclusion:} Loops are a symptom of insufficient computation budget, not depth per se.

\subsection{E5: Generalization to Parity}

Parity task ("light on/off"), $n_{\text{ops}} \in \{4\ldots 48\}$, 10 seeds:
\begin{itemize}
    \item Winding $\sim n_{\text{ops}}$: $\rho = 0.162$, $p = 0.22$ (n.s.).
    \item Steps-to-settle $\sim n_{\text{ops}}$: $\rho = 0.615$, $p < 1e-6$.
\end{itemize}
The same signature generalizes beyond arithmetic to other counting tasks, confirming the robustness of H3.

\section{Reproduction of Two-Scale Dynamics}

Applying Pappone et al. (2025)'s two-scale metrics to synthetic data (depths 2--5), we find:

\begin{table}[H]
\centering
\caption{Two-scale dynamics on synthetic data (depths 2--5)}
\begin{tabular}{cccc}
\toprule
Depth & Mean Acceleration & Mean Orthogonality & Exits (\%) \\
\midrule
2 & $4.89 \pm 0.02$ & $-0.497 \pm 0.002$ & 50\% \\
3 & $5.84 \pm 0.01$ & $-0.498 \pm 0.001$ & 50\% \\
4 & $6.70 \pm 0.02$ & $-0.498 \pm 0.001$ & 50\% \\
5 & $7.45 \pm 0.02$ & $-0.498 \pm 0.002$ & 50\% \\
\bottomrule
\end{tabular}
\end{table}

Acceleration grows with depth ($4.89 \rightarrow 7.45$, $\rho = 0.997$). Orthogonality is stable ($\sim -0.5$). However, during reproduction on real FineWeb text data, we obtained results that \textbf{fundamentally differ} from Pappone et al. \cite{pappone2025two}.

\begin{table}[H]
\centering
\caption{Comparison with Pappone et al. (2025) on FineWeb}
\begin{tabular}{lcc}
\toprule
Aspect & Pappone et al. (2025) & Our Results \\
\midrule
Data & FineWeb (real) & FineWeb (real) \\
Dynamics type & Spiral ($\cos \approx 0.5$--0.65) & Oscillatory ($\cos \approx -0.687$) \\
Orthogonality & Achieved & Achieved \\
Step normalization & Not applied & Applied (critically important) \\
Correlation with norm & Not addressed & Resolved ($\text{corr} = -0.0436$) \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[H]
\centering
\caption{Metrics on real FineWeb data}
\begin{tabular}{lc}
\toprule
Parameter & Value \\
\midrule
Orthogonality ($\cos$ angle) & $-0.6870 \pm 0.0993$ \\
Angle between steps & $133.9^\circ$ \\
Acceleration & $1.8360 \pm 0.0548$ \\
Step norm & $13.7350 \pm 3.1820$ \\
Two-hit exit step & 5 \\
\bottomrule
\end{tabular}
\end{table}

Without step normalization, orthogonality is not stably achieved. Only after applying normalization do we obtain stable orthogonality, revealing oscillatory dynamics ($\cos \approx -0.687$) instead of spiral dynamics. This indicates the original conclusions may be methodological artifacts.

\section{Methodological Retractions and Amendments}

We report the claims this project withdrew alongside the ones it kept, because several of the withdrawals were produced by controls we had registered in advance and are therefore evidence about the method rather than only about the errors. In four cases a result was killed by its own pre-registered gate: a matched-pair geometry claim did not survive the norm control it had itself specified; a spectral run was refused by its own consistency check (a 48-unroll tail failing its P1 check due to a damping/rotation confound); a behavioural null turned out to have been computed on a variable that took the same value in 432 of 432 orbits; and a scoring comparison was found to be measuring the parser rather than the model. Additionally, we retract our earlier claim that D162's protocol is refuted as the explanation, as prompt format was the whole story and was never varied. Furthermore, D192 first attributed our failure-to-reproduce to our own detector being period-6-blind, but D200 shows that was wrong: there is almost no non-DC power to distribute.

\section{Limitations}

\begin{itemize}
    \item \textbf{Answer-token only.} We examined only the final token; Geiping et al. observed orbits on question/digit tokens.
    \item \textbf{Small sample sizes.} Insufficient power to detect rare phenomena (0.02--2.8\% per Blayney).
    \item \textbf{Proxy metrics.} 2D-PCA winding (sign-arbitrary); steps-to-settle conflates small steps with convergence.
    \item \textbf{Single initialization seed.} Most runs use one seed; apparent signals collapsed under multi-seed testing.
    \item \textbf{Geiping regime not reproduced.} We did not use their system prompt or $r=128$.
    \item \textbf{Spectral radius $\rho(J)$ not computed.} This is the key metric for H3; we used convergence proxies.
\end{itemize}

\section*{Acknowledgments}
The authors thank curator S. Barannikov and mentor N. Kurdyukov for valuable discussions and guidance. We formally thank the SMILES-26 organizers and Yandex Cloud for compute grants. Code and data are available at \url{https://github.com/Shtirmann/Geometry-of-Reasoning-Trajectories}.

\begin{thebibliography}{9}

\bibitem{geiping2025scaling}
Geiping, J., et al. (2025). "Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach." \textit{NeurIPS 2025}. arXiv:2502.05171.

\bibitem{blayney2026mechanistic}
Blayney, E., et al. (2026). "A Mechanistic Analysis of Looped Reasoning Language Models." arXiv:2604.11791.

\bibitem{pappone2025two}
Pappone, L., et al. (2025). "Two-Scale Latent Dynamics for Recurrent-Depth Transformers." \textit{NeurIPS 2025}. arXiv:2509.23314.

\bibitem{merrill2024illusion}
Merrill, W., Petty, J., \& Sabharwal, A. (2024). "The Illusion of State in State-Space Models." arXiv:2404.08819.

\bibitem{grazzi2024unlocking}
Grazzi, R., et al. (2024). "Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues." arXiv:2411.12537.

\bibitem{dehghani2019ut}
Dehghani, M., et al. (2019). "Universal Transformers." \textit{ICLR 2019}. arXiv:1807.03819.

\end{thebibliography}

\newpage
\section*{Appendix: Author Contributions}

\begin{table}[H]
\centering
\caption{Contribution matrix}
\begin{tabular}{p{3.5cm}p{3.2cm}p{3.2cm}p{3.2cm}}
\toprule
Component & Alexandr Shiianov & David Sverdlov & Arsenii Varaksin \\
\midrule
Formatting confound & -- & -- & \checkmark \\
Rotation regime & -- & -- & \checkmark \\
Timing hinge & -- & -- & \checkmark \\
Extraction pipeline & \checkmark & -- & -- \\
Winding number & \checkmark & -- & -- \\
Steps-to-settle & \checkmark & -- & -- \\
PARARULE/Counting & \checkmark & -- & -- \\
Length-matched ctrl & \checkmark & -- & -- \\
Two-scale dynamics & -- & \checkmark & -- \\
FineWeb repro & -- & \checkmark & -- \\
\bottomrule
\end{tabular}
\end{table}

\subsection*{Detailed Contributions}

\textbf{Alexandr Shiianov (40\% of work):} Developed trajectory extraction infrastructure; implemented MVP metrics (winding, steps-to-settle); conducted experiments E1--E5 across PARARULE-Plus, counting, length-matched controls, phase maps, and generalization; implemented persistent homology.

\textbf{David Sverdlov (40\% of work):} Implemented two-scale dynamics metrics; conducted reproduction of orthogonality on real FineWeb data; demonstrated that without step normalization, orthogonality is not achieved; validated acceleration as stronger depth indicator than winding.

\textbf{Arsenii Varaksin (20\% of work):} Discovered the formatting evaluation confound; defined and validated the scalar rotation statistic; established the timing hinge separating regime from answer computation; executed causal embedding steering experiments.

\begin{russian}
\section*{Р Р…Р РЋР вЂњР В Р РЋР В Р РЋР В Р РЋР В РЎвЂ”Р Р…Р В РЎвЂ”}
\textbf{Р РЋР В РЎвЂ”Р В Р РЋР В Р РЋ} \\
\textit{Р РЋР В РЎвЂ”Р В Р РЋР В Р РЋР В Р РЋ}

Р РЋР В РЎвЂ”Р В Р РЋР В Р РЋР В Р РЋ
\end{russian}

\end{document}
"""

with open('/Users/a2mogus/build-projs/barannikov-work/Geometry-of-Reasoning-Trajectories/docs/submission/build/final_submission.tex', 'w', encoding='cp1251', errors='replace') as f:
    f.write(latex_content)
