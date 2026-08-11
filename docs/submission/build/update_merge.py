import re

with open('merged_paper.tex', 'r', encoding='cp1251') as f:
    text = f.read()

appendices = r"""\appendix

\section{Numerical Summary}

\begin{table}[H]
\centering
\small
\caption{Principal quantitative results. Inference-only on the pinned revision.}
\begin{tabularx}{\linewidth}{@{}lX@{}}
\toprule
\textbf{Quantity} & \textbf{Value} \\
\midrule
Regime switch by one instruction word & 36/36 rotating vs 0/36, token count fixed at 53 \\
Semantic control & neighbours of \emph{symbol} 0/144; of \emph{element} 96/120 \\
Form-matched control & 72/72 rotating \\
Threshold reproduces binary label & 691/696; 688/696 held out \\
Embedding-row edit flips regime & gates bit-exact at $0.000\mathrm{e}{+}00$ \\
Regime by position & rotating prompts 62.0\% of positions; settling 0.000 \\
Mid-trajectory parameter swap & 62/64 switches take \\
Relaxation asymmetry & entering 15.5 unrolls, leaving 1.0 \\
Predicted from $\rho \approx 0.83$ & 16.1 unrolls \\
First-token vs produced answer & 0.125 against 0.781 \\
Chance rate for containment & 0.042--0.152 \\
\emph{The} as opening token & 39.4\% of 1260 generations \\
Gold's worst final rank (feasible tasks) & 35, over 2720 draws \\
ARC-Easy, published & 0.699 at $r = 32$ \\
ARC-Easy, matched protocol & 0.658 \\
ARC-Easy, five-shot letter argmax & 0.723 ($p = 3.35\mathrm{e}{-}07$) \\
Two-shot effect on oracle accuracy & $+0.221$ ($p = 9.3\mathrm{e}{-}09$) \\
Irrelevant padding, start rank & 31 $\rightarrow$ 329 ($p = 2.9\mathrm{e}{-}09$) \\
Answer position vs prose positions & 7.89 against 3.65 unrolls ($p = 0.0004$) \\
Decision window (unrolls 0--11) & rotating $0.2292$ vs settling $0.2296$; separation $0.0004$ \\
Tail window & $0.862$ vs $0.298$; separation $0.565$ \\
Regime onset & between window starts 12 and 16 \\
Steering, geometry arm & $|\Delta R| \le 0.0065$; 0 threshold crossings in 132 conditions \\
Probe control (instrument null) & 0.690 vs 0.600 baseline; correct statistic 0.980 \\
\bottomrule
\end{tabularx}
\end{table}

\section{Index of Experiments}

Each result in the body traces to one run. Every run is inference-only on the pinned revision,
with the initial state seeded before each forward pass so that repeated forwards are comparable.

\begin{table}[H]
\centering
\small
\caption{The experiments behind each section, with their designs.}
\begin{tabularx}{\linewidth}{@{}llX@{}}
\toprule
\textbf{\S} & \textbf{Run} & \textbf{Design} \\
\midrule
3.1 & threshold & re-analysis of 696 orbits over three banks, no GPU \\
3.2 & word swap & 18 sequences $\times$ 3 instructions $\times$ 2 markers, token count fixed at 53 \\
3.2 & noun sweep & 16 nouns $\times$ 12 shared sequences $\times$ 2 markers (384 orbits) \\
3.3 & embedding swap & 252 orbits; interpolation of one \texttt{wte} row, both endpoints gated \\
3.4 & position sweep & 21 prompts, every token position, 64 unrolls \\
3.5 & hysteresis & 64 mid-trajectory parameter swaps at 8 switch points, both directions \\
4 & generation scoring & 128 generations at $r = 32$ and $48$, with measured chance rates \\
4 & depth bank & 1260 generations, 21 families, 5 depths (re-analysis, no GPU) \\
4 & benchmark arms & 120 items, 4 scoring/prompt arms; plus a 101-item paired shot comparison \\
4 & format sweep & 24 items per arm, instruction wording varied at fixed content \\
4 & demonstration & paired 0- vs 2- vs 5-shot on the census families \\
4 & steering & 132 conditions spanning both sides of the regime threshold \\
4 & length control & zero-shot padded with neutral prose to the 5-shot token length \\
4 & answer position & 36 generations, every generated position, 48 unrolls \\
5 & window split & 48 prompts; the same statistic on the decision window and the tail \\
8 & probe control & 204 banked arrays over 51 nouns, leave-one-noun-out \\
\bottomrule
\end{tabularx}
\end{table}

\section{Withdrawn and Amended Claims}

Ten ledger rows carry an explicit withdrawal or amendment marker. Every entry below states what
fell \emph{and} what survived, because in each case something did.

\paragraph{Inherited from earlier work.} \textbf{D34} and \textbf{D36} are superseded by D50: the
direction they measured is 97\% the current-token contrast, so the window statistic they rest on
is an artefact --- the register claim itself survives, the stated mechanism does not.
\textbf{D43} is retracted; it held that the contraction rate predicts how deep the model can
usefully think, and direct measurement flipped the verdict. \textbf{D51} is subsumed by a
same-day rerun on more checkpoints, though its caveat about the early jump was confirmed.

\paragraph{The regime line.} \textbf{D137/D138} are replaced by the threshold result: their
``gapped, therefore two regimes'' test used a uniform null, which any clustered distribution
beats, so it never tested the claim being made; checked against the labels, the gap splits the
settling population. The replacement is stronger, reproducing the binary label on 691 of 696
orbits. \textbf{D135/D136} are superseded because the probe class they used cannot resolve the
question; what survives is five held-out replications. \textbf{D164} was amended by
its own registered norm control, which the decisive matched pair did not survive --- the
measurements replicate bit-for-bit, the interpretation is what failed.

\paragraph{The evaluation line.} \textbf{D148}'s correctness half is withdrawn as vacuous:
\texttt{correct} was False in 432 of 432 orbits, so there was no variance for the regime to
change. Re-tested on an outcome that varies, the conclusion returned at 1 of 18 paired units.
\textbf{D162}'s headline refuted ``protocol'' after varying only the scoring rule at fixed
prompt; prompt format was the whole story and was never varied. \textbf{D193}'s parsed-exact
column measured our parser rather than the model, reading 14$\times$ low on the harness arm; on
the repaired metric its conclusion survives unchanged.

\paragraph{Narrowed rather than withdrawn.} Three interpretations were tightened without the
underlying row falling: arithmetic families are not ``uncertain about the answer'' but compute in
prose; depth tracks where the answer starts rather than the kind of operation; and the steering
nulls are bounded to the last-position injection site rather than to orthogonality of directions.

\paragraph{Refused by their own gates.} Two runs never became claims at all: a spectral pass whose
tail length disagreed with the statistic it was validating against, and a reasoning arm that hit
its wall-clock budget and banked cleanly without executing.

\section{Why a Statistic Rather Than a Projection}

The published account of this model's latent dynamics is established by projecting trajectories
onto their leading principal components and reading the pictures. That practice cannot support the
claims we needed to make, for four reasons.

\emph{A projection is fitted to the data it judges.} Two components of a 5280-dimensional
trajectory retain a small fraction of the variance directions, and which fraction is chosen by the
trajectory itself. Whether a loop appears is therefore a property of the basis as much as of the
dynamics.

\emph{Damping is confounded with rotation.} A monotone settle along a curved path renders as an
arc, and an arc read as a partial orbit cannot be falsified by looking harder at the same picture.

\emph{A projection cannot support a negative.} ``This prompt does not orbit'' does not follow from
a picture in which no orbit is evident. Two of our central results are negatives --- that the
authors' own illustrative prompt does not orbit at any period the tail resolves, and that settling
prompts rotate at none of their token positions --- and neither is available by inspection.

\emph{A projection cannot support a threshold.} Without one, ``rotating'' is a judgement made per
figure: no population statement, no held-out validation, no cross-prompt comparison. The threshold
$R > 0.6677$ is what licenses 691 of 696 and 688 held out.

"""

text = text.replace(r"\begin{thebibliography}{9}", appendices + "\n" + r"\begin{thebibliography}{9}")

with open('merged_paper.tex', 'w', encoding='cp1251') as f:
    f.write(text)

